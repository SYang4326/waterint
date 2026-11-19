"""GPU backend of hb-correlation: CUDA streaming kernel in a separate process (built with nvcc on first use).

Same interface as analysis.dynamics.hb_correlation_core.HBCorrelationKernel; frames are streamed to the kernel's
stdin. Imported only when the gpu backend is requested; the core package never needs CUDA.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


GPU_SOURCE = Path(__file__).resolve().parent / "hb_correlation_gpu.cu"


def gpu_binary_path() -> Path:
    import hashlib
    import os
    from waterint.backends.cpu.native import cache_dir
    tag = hashlib.sha256(GPU_SOURCE.read_bytes() + os.environ.get("WATERINT_CUDA_ARCH", "sm_90").encode()).hexdigest()[:16]
    return cache_dir() / f"hb_correlation_gpu_{tag}"


def build_gpu_kernel(arch: str | None = None, nvcc: str | None = None) -> Path:
    """nvcc -O3 -std=c++17 -fmad=false -arch=<arch> (default WATERINT_CUDA_ARCH or sm_90, H100/H200)."""
    import os
    import shutil
    import subprocess
    nvcc = nvcc or os.environ.get("NVCC") or shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found: load a CUDA module or set NVCC (GPU backend of hb-correlation)")
    arch = arch or os.environ.get("WATERINT_CUDA_ARCH", "sm_90")
    out = gpu_binary_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([nvcc, "-O3", "-std=c++17", "-fmad=false", f"-arch={arch}", str(GPU_SOURCE), "-o", str(out), "-lpthread"],
                   check=True, capture_output=True, text=True)
    return out


class HBCorrelationGPUProcess:
    """Same interface as HBCorrelationKernel; frames are streamed to the CUDA kernel's stdin (record per frame:
    float64 z_ref, z_ref2, O/H coordinates), outputs are the same raw files. -fmad=false keeps the C_PT / C_HB / FES
    counts identical to the CPU kernel; orientation may differ by a few counts in 1e7 (GPU acos last bit)."""

    def __init__(self, geom, n_frames: int, lx: float, ly: float, out_dir: Path, *, window: int = 100000,
                 block: int = 250000, msd_stride: int = 10, msd_maxlag: int = 250000, binary: str | Path | None = None) -> None:
        import subprocess
        self.geom = geom
        self.out = Path(out_dir)
        gdir = self.out / "geom"
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "atom_ids.bin").write_bytes(geom.ids.astype("<i4").tobytes())
        (gdir / "atom_types.bin").write_bytes(geom.types.astype("<i2").tobytes())
        (gdir / "lattice_O_ids.txt").write_text("".join(f"{i}\n" for i in geom.ids[geom.lattice.astype(bool)]))
        exe = Path(binary) if binary else gpu_binary_path()
        if not exe.is_file():
            exe = build_gpu_kernel() if not binary else exe
        self.log = open(self.out / "kernel.log", "w")
        self.proc = subprocess.Popen([str(exe), str(gdir), str(self.out), str(int(n_frames)), repr(float(lx)), repr(float(ly)),
                                      str(window), str(block), str(msd_stride), str(msd_maxlag)],
                                     stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT)
        self.threads = "gpu"

    def push(self, pos: np.ndarray) -> None:
        zr, zr2 = self.geom.zrefs(pos)
        rec = np.empty((pos.shape[0], 2 + 3 * len(self.geom.oh)), dtype="<f8")
        rec[:, 0], rec[:, 1] = zr, zr2
        rec[:, 2:] = pos[:, self.geom.oh, :].reshape(pos.shape[0], -1)
        try:
            self.proc.stdin.write(rec.tobytes())
        except BrokenPipeError:
            self.proc.wait()
            raise RuntimeError(f"GPU kernel exited early: {(self.out / 'kernel.log').read_text()[-2000:]}")

    def finish(self, out_dir: Path, read_seconds: float = 0.0) -> None:
        self.proc.stdin.close()
        rc = self.proc.wait()
        self.log.close()
        if rc != 0:
            raise RuntimeError(f"GPU kernel rc {rc}: {(self.out / 'kernel.log').read_text()[-2000:]}")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
