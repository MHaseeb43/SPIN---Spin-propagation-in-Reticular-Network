"""
Lazy reading of binarised OME-TIFF stacks.

The stacks are large (up to ~100 Z-layers of ~8000 × 8000 px), so we open
them through zarr/dask and pull in one layer at a time.
"""

import numpy as np
import tifffile
import dask.array as da


def open_stack(path: str):
    """
    Open a TIFF stack lazily.

    Returns (image_lazy, store). The caller is responsible for calling
    `store.close()` when done. 2-D files come back with shape (H, W).
    """
    store = tifffile.imread(path, aszarr=True)
    image_lazy = da.from_zarr(store)
    print(f"  Stack shape: {image_lazy.shape}, dtype: {image_lazy.dtype}")
    return image_lazy, store


def load_layer(image_lazy, layer: int) -> np.ndarray:
    """Load one Z-layer as a binary uint8 (0/1) array."""
    image_np = image_lazy[layer].compute() if image_lazy.ndim == 3 else image_lazy.compute()
    return (image_np > 0).astype(np.uint8)
