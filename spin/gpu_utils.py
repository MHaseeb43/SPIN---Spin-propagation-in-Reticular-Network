"""
Optional GPU support.

CuPy (max projection) and cuCIM (skeletonisation) are used when they are
available; everything else in the pipeline runs on the CPU. Import GPU
libraries through this module so the code keeps working on machines
without a GPU.

Note: cuCIM is not available through pip on Windows, so there the
skeletonisation always runs on the CPU.
"""

from __future__ import annotations
import warnings
import numpy as np

# CuPy warns at import time when CUDA_PATH is not set, even if it works fine.
warnings.filterwarnings("ignore", category=UserWarning, module="cupy")

try:
    import cupy as cp
    # cp.zeros() works even without NVRTC; the OR forces an actual kernel
    # compilation, so this really tells us whether CuPy is usable.
    _a = cp.zeros(1, dtype=cp.bool_)
    _ = _a | _a
    del _a, _
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False

try:
    import cucim.skimage.morphology  # noqa: F401
    _CUCIM_AVAILABLE = True
except Exception:
    _CUCIM_AVAILABLE = False


def gpu_available() -> bool:
    """True if CuPy can run kernels on a CUDA GPU."""
    return _CUPY_AVAILABLE


def cucim_available() -> bool:
    """True if cuCIM (GPU skeletonisation) is installed."""
    return _CUCIM_AVAILABLE


def to_gpu(arr: np.ndarray):
    """Move an array to the GPU, or return it unchanged if there is no GPU."""
    return cp.asarray(arr) if _CUPY_AVAILABLE else arr


def to_cpu(arr) -> np.ndarray:
    """Bring an array back to NumPy (no-op for NumPy arrays)."""
    if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)
