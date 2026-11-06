"""
Skeletonisation: thin the binary conduit mask down to 1-px-wide centre lines.

Runs on the GPU with cuCIM when it is installed (RAPIDS, via conda) and
falls back to scikit-image on the CPU otherwise.
"""

import numpy as np
import tifffile
from spin.gpu_utils import cucim_available, to_gpu, to_cpu


def compute_skeleton(binary: np.ndarray, save_path: str | None = None,
                     method: str | None = "lee") -> np.ndarray:
    """
    Skeletonise a 2-D binary image and return a boolean skeleton.

    method : 'lee' (default), 'zhang' or None
        Thinning algorithm passed on to skeletonize(). On our data Lee (1994)
        gives fewer spurious side branches than Zhang, which is what
        scikit-image would pick by default for 2-D input (method=None).
    save_path : optional path to also write the skeleton as a TIFF.
    """
    if cucim_available():
        import cucim.skimage.morphology as cucim_morph
        skeleton = to_cpu(cucim_morph.skeletonize(to_gpu(binary.astype(bool)), method=method))
        backend = "cuCIM GPU"
    else:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(binary, method=method)
        backend = "skimage CPU"
    skeleton = skeleton.astype(bool)

    print(f"  Skeleton pixels: {skeleton.sum():,}  [{backend}, method={method or 'auto'}]")

    if save_path:
        tifffile.imwrite(save_path, (skeleton * 255).astype(np.uint8))
        print(f"  Skeleton saved -> {save_path}")

    return skeleton
