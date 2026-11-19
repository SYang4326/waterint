"""Optional GPU (CUDA) kernels.

Nothing here is imported by the core package unless a GPU backend is requested or ``auto`` asks for a
kernel registered with ``prefer_gpu=True``. GPU kernels are compiled on demand with nvcc into the
WaterInt cache directory; when nvcc or a CUDA device is missing, ``available()`` is False and every
kernel falls back to its CPU / Python implementation.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def available() -> bool:
    if os.environ.get("WATERINT_NO_GPU", "0") == "1":
        return False
    if shutil.which("nvcc") is None and not os.environ.get("CUDA_HOME"):
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and "GPU" in out.stdout
