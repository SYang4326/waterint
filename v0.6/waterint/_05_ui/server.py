from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from waterint._05_ui.core import RUNNERS, artifact, collect_artifacts, parse_ui_config, public_config, resolve_base_dir, resolve_output_dir

import yaml


@dataclass
class Job:
    id: str
    module: str
    status: str = "queued"
    message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    config_path: str | None = None
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = time.time()


def run_ui_server(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True) -> None:
    store = JobStore()
    handler = _make_handler(store, Path.cwd())
    server = ThreadingHTTPServer((host, port), handler)
    actual_host, actual_port = server.server_address[:2]
    url_host = host if host not in {"", "0.0.0.0"} else actual_host
    url = f"http://{url_host}:{actual_port}"
    print(f"WaterInt UI: {url}")
    print("Press Ctrl-C to stop the UI server.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping WaterInt UI.")
    finally:
        server.server_close()


def _make_handler(store: JobStore, cwd: Path):
    static_dir = Path(__file__).resolve().parent / "static"

    class WaterIntUIHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[waterint-ui] {self.address_string()} - {fmt % args}")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                return self._send_file(static_dir / "index.html")
            if parsed.path.startswith("/static/"):
                static_path = _static_path(static_dir, parsed.path.removeprefix("/static/"))
                if static_path is None:
                    return self._send_json({"error": "Not found"}, status=404)
                return self._send_file(static_path)
            if parsed.path == "/api/status":
                return self._send_json({"ok": True, "cwd": str(cwd)})
            if parsed.path.startswith("/api/jobs/"):
                parts = parsed.path.strip("/").split("/")
                if len(parts) == 3:
                    return self._send_job(parts[2])
                if len(parts) == 5 and parts[3] == "artifacts":
                    return self._send_artifact(parts[2], parts[4])
            return self._send_json({"error": "Not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/run":
                return self._start_job()
            return self._send_json({"error": "Not found"}, status=404)

        def _start_job(self) -> None:
            try:
                payload = self._read_json()
                module = str(payload.get("module", "")).strip()
                if module not in RUNNERS:
                    raise ValueError(f"Unknown module: {module}")
                config_yaml = str(payload.get("config_yaml", "")).strip()
                base_dir = resolve_base_dir(payload.get("base_dir"), cwd)
                config = parse_ui_config(config_yaml, base_dir)
                job_id = uuid.uuid4().hex[:12]
                job = Job(id=job_id, module=module)
                store.add(job)
                thread = threading.Thread(
                    target=_run_job,
                    args=(store, job_id, module, config, base_dir),
                    daemon=True,
                )
                thread.start()
                self._send_json({"job_id": job_id, "status": job.status})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def _send_job(self, job_id: str) -> None:
            job = store.get(job_id)
            if job is None:
                return self._send_json({"error": "Unknown job"}, status=404)
            self._send_json(_job_payload(job))

        def _send_artifact(self, job_id: str, artifact_id: str) -> None:
            job = store.get(job_id)
            if job is None:
                return self._send_json({"error": "Unknown job"}, status=404)
            match = next((item for item in job.artifacts if item["id"] == artifact_id), None)
            if match is None:
                return self._send_json({"error": "Unknown artifact"}, status=404)
            return self._send_file(Path(match["path"]))

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw or "{}")
            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object.")
            return data

        def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                return self._send_json({"error": "File not found"}, status=404)
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return WaterIntUIHandler


def parse_ui_config(config_yaml: str, base_dir: Path) -> dict[str, Any]:
    if not config_yaml:
        raise ValueError("Config YAML is empty.")
    config = yaml.safe_load(config_yaml)
    if not isinstance(config, dict):
        raise ValueError("Config YAML must define a mapping.")
    config["_config_dir"] = str(base_dir.resolve())
    config["_config_path"] = str((base_dir / "waterint-ui-config.yaml").resolve())
    return config


def _run_job(store: JobStore, job_id: str, module: str, config: dict[str, Any], base_dir: Path) -> None:
    try:
        store.update(job_id, status="running", message="Running analysis")
        output_cfg = config.setdefault("output", {})
        if not isinstance(output_cfg, dict):
            raise ValueError("output must be a mapping.")
        outdir = resolve_output_dir(config, base_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        config_path = outdir / "waterint_ui_config.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(public_config(config), handle, sort_keys=False)
        store.update(job_id, config_path=str(config_path), message="Config written, executing workflow")
        result = RUNNERS[module](config)
        artifacts = collect_artifacts(result)
        artifacts.insert(0, artifact("config", config_path))
        store.update(job_id, status="complete", message="Analysis complete", artifacts=artifacts)
    except Exception as exc:
        store.update(job_id, status="failed", message="Analysis failed", error=str(exc))


def _static_path(static_dir: Path, relative_path: str) -> Path | None:
    root = static_dir.resolve()
    candidate = (static_dir / relative_path).resolve()
    if candidate == root or root not in candidate.parents:
        return None
    return candidate


def _job_payload(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "module": job.module,
        "status": job.status,
        "message": job.message,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "config_path": job.config_path,
        "artifacts": job.artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="waterint ui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser automatically.")
    args = parser.parse_args(argv)
    run_ui_server(args.host, args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
