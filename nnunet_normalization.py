"""
PURPOSE
    Implements a robust, percentile-clipped z-score computed inside the brain mask, as an
    alternative to nnU-Net's default per-image z-score.

USAGE
    1. Place this module on the PYTHONPATH.
    2. In nnUNet_preprocessed/<Dataset>/nnUNetPlans.json, set for the target configuration:
           "normalization_schemes": ["ClippedZScoreNormalization"]
           "use_mask_for_norm": [true]
       nnU-Net resolves the class by name from nnunetv2.preprocessing.normalization, so the
       class must be registered there (copy the file into that package, or import it from the
       package __init__).
    3. Re-run nnUNetv2_preprocess for that plan.

PIPELINE POSITION
    raw -> [offline cache: reorient + optional N4] -> nnU-Net crop -> THIS -> resample -> patch
"""

from __future__ import annotations

import numpy as np
from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
    ImageNormalization,
)


class ClippedZScoreNormalization(ImageNormalization):
    """
    Percentile-clipped z-score, computed inside the brain mask.

    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm must be True: nnU-Net relies on
    this flag to know that the background stays at exactly zero after normalization, which is
    what allows the nonzero region to remain a valid brain mask downstream.
    """

    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm = True

    # Clipping bounds, in percentiles of the foreground intensity distribution. 0.5 / 99.5
    # mirrors the bounds nnU-Net uses for CT, and is a conservative starting point: it removes
    # the extreme tail without touching the tissue range.
    LOWER_PERCENTILE = 0.5
    UPPER_PERCENTILE = 99.5

    def run(self, image: np.ndarray, seg: np.ndarray | None = None) -> np.ndarray:
        """
        Normalize one image channel.

        Inputs:
            image: single-channel array (nnU-Net calls this once per channel)
            seg:   segmentation array; when use_mask_for_norm is set, nnU-Net encodes the
                   nonzero region as seg >= 0 and the background as seg == -1
        Outputs:
            normalized float32 array, background exactly zero when a mask is used
        """
        image = image.astype(np.float32, copy=True)

        use_mask = (
            self.use_mask_for_norm is not None
            and self.use_mask_for_norm
            and seg is not None
        )
        mask = (seg >= 0) if use_mask else np.ones(image.shape, dtype=bool)

        fg = image[mask]
        if fg.size == 0:
            # Degenerate case (empty brain mask). Surfaced by the QA pass; here the image is
            # returned zeroed rather than producing NaNs that would poison the batch.
            return np.zeros_like(image)

        lo, hi = np.percentile(fg, [self.LOWER_PERCENTILE, self.UPPER_PERCENTILE])
        np.clip(image, lo, hi, out=image)

        # Statistics are recomputed AFTER clipping. Computing them before would let the very
        # outliers the clip is meant to remove keep inflating the standard deviation, which is
        # the failure mode this scheme exists to fix.
        fg = image[mask]
        mean = float(fg.mean())
        std = float(fg.std())
        image = (image - mean) / max(std, 1e-8)

        if use_mask:
            image[~mask] = 0.0
        return image