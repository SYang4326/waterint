"""Kernel registry: every heavy kernel has a Python reference and optional CPU (C++) / GPU (CUDA) versions.

    @register("density.histogram", "python")
    def _hist_py(values, edges): ...

    backend, fn = select("density.histogram", requested="auto")

``auto`` picks gpu (only for kernels registered with ``prefer_gpu=True`` and when a GPU is usable),
then cpu (when the native library loads), then python. An explicit request that cannot be met raises.
The core package never imports CUDA: GPU kernels live in ``waterint.backends.gpu`` and are imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

BACKENDS = ("python", "cpu", "gpu")
ALIASES = {"cpp": "cpu", "c++": "cpu", "native": "cpu", "cuda": "gpu"}


@dataclass
class Kernel:
    name: str
    impls: dict[str, Callable[..., Any]] = field(default_factory=dict)
    prefer_gpu: bool = False


_KERNELS: dict[str, Kernel] = {}


def register(name: str, backend: str, *, prefer_gpu: bool = False) -> Callable[[Callable], Callable]:
    backend = normalize(backend)

    def deco(fn: Callable) -> Callable:
        kernel = _KERNELS.setdefault(name, Kernel(name))
        kernel.impls[backend] = fn
        kernel.prefer_gpu = kernel.prefer_gpu or prefer_gpu
        return fn

    return deco


def normalize(requested: str | None) -> str:
    value = str(requested or "auto").lower()
    value = ALIASES.get(value, value)
    if value not in (*BACKENDS, "auto"):
        raise ValueError(f"backend must be auto, python, cpu (cpp), or gpu; got {requested!r}.")
    return value


def cpu_available() -> bool:
    from waterint.backends.cpu.native import native_library
    return native_library() is not None


def gpu_available() -> bool:
    try:
        from waterint.backends import gpu
    except Exception:
        return False
    return bool(gpu.available())


def _usable(backend: str) -> bool:
    if backend == "python":
        return True
    if backend == "cpu":
        return cpu_available()
    return gpu_available()


def available(name: str) -> list[str]:
    kernel = _get(name)
    return [b for b in BACKENDS if b in kernel.impls and _usable(b)]


def select(name: str, requested: str | None = "auto") -> tuple[str, Callable[..., Any]]:
    kernel = _get(name)
    want = normalize(requested)
    if want != "auto":
        if want not in kernel.impls:
            raise RuntimeError(f"Kernel {name!r} has no {want} implementation (has: {sorted(kernel.impls)}).")
        if not _usable(want):
            raise RuntimeError(f"Kernel {name!r}: the {want} backend was requested but is not available "
                               f"({backend_status(want)}).")
        return want, kernel.impls[want]
    order = ("gpu", "cpu", "python") if kernel.prefer_gpu else ("cpu", "python")
    for backend in order:
        if backend in kernel.impls and _usable(backend):
            return backend, kernel.impls[backend]
    raise RuntimeError(f"Kernel {name!r} has no usable implementation.")


def backend_status(backend: str) -> str:
    if backend == "cpu":
        from waterint.backends.cpu.native import native_status
        status = native_status()
        return "available" if status["available"] else f"unavailable: {status['last_error'] or 'no compiler'}"
    if backend == "gpu":
        return "available" if gpu_available() else "unavailable (no CUDA GPU / nvcc, or no GPU kernels built)"
    return "available"


def describe() -> dict[str, list[str]]:
    """Kernel name -> registered backends (used by docs and `waterint backends`)."""
    return {name: [b for b in BACKENDS if b in k.impls] for name, k in sorted(_KERNELS.items())}


def _get(name: str) -> Kernel:
    if name not in _KERNELS:
        raise KeyError(f"Unknown kernel {name!r}. Registered: {sorted(_KERNELS)}")
    return _KERNELS[name]
