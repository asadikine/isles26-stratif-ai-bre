"""
ISLES'26 - Canonical preprocessing layer (shared contract).

PURPOSE
    This module is the SINGLE source of truth for the image transformations applied to an
    ISLES'26 case. It is imported by two callers that must never diverge:

        build_training_cache.py   offline, over the K training folds -> writes an nnU-Net
                                  raw dataset on disk (avoids on-the-fly preprocessing when
                                  training several models on the same base)
        inference entrypoint      inside the Docker container, per case, at submission time

    The holdout is deliberately NOT cached: it is processed through this module at inference
    time, which makes it a genuine rehearsal of the container.

PIPELINE / DATA FLOW (per case)
    raw T1w NIfTI (native space, already skull-stripped by SynthStrip)
        |
        v
    [1] QA / integrity checks: finite values, image-mask affine agreement, mask domain,
        skull-strip sanity. Failures are reported, never silently repaired.
        |
        v
    [2] Canonical reorientation to RAS. Axis permutation and flips only: lossless and
        exactly invertible. The original affine and shape are retained so that the predicted
        mask can be written back on the exact input voxel grid (native-space requirement of
        the challenge evaluation).
        |
        v
    [3] Optional N4ITK bias field correction, restricted to the brain mask (image > 0).
        Treated as an ABLATION VARIANT, not as an established step: published ablations
        report no CNN benefit from bias-field correction, and the reference nnU-Net results
        on ATLAS v2.0 T1w were obtained without it. Two cache variants are therefore built.
        |
        v
    [4] Cast: image float32, mask uint8.
        |
        v
    canonical array + CanonicalGeometry (carries the inverse transform)

WHAT THIS MODULE DELIBERATELY DOES NOT DO
    - Skull-stripping: already performed by the organizers.
    - Resampling to a target spacing: performed once by nnU-Net. Doing it here as well would
      cost a second interpolation, which is paid on exactly the structures that decide the
      ranking (small multifocal lesions -> Detection F1, lesion count).
    - Intensity clipping / z-score: per-image, cheap, and a hyperparameter. It belongs to the
      nnU-Net normalization scheme (see nnunet_normalization.py), not to a frozen cache.
    - Any modification of the ground-truth mask: the ranked metrics include connected-
      component count and empty-GT cases. Cleaning the target corrupts the objective.

PREREQUISITES
    nibabel, numpy, SimpleITK (only when apply_n4 is True).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.orientations import io_orientation, ornt_transform

# Fraction of exactly-zero voxels below which a volume is unlikely to be skull-stripped.
# SynthStrip zeroes the background exactly, and a brain occupies roughly 20-40 percent of a
# T1 field of view, so a compliant volume sits well above this floor.
MIN_BACKGROUND_FRACTION = 0.30

# Sentinel used in the conditioning vector when a covariate is absent. The paired *_missing
# flag is what the network actually reads; the sentinel only keeps the tensor well defined.
MISSING_SENTINEL = 0.0


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CanonicalGeometry:
    """
    Everything required to map a canonical (RAS) array back onto the input voxel grid.

    Attributes:
        original_affine:  affine of the input NIfTI, verbatim
        original_shape:   shape of the input NIfTI, verbatim
        canonical_affine: affine after reorientation to RAS
        zooms:            voxel spacing (mm) in the canonical frame, x/y/z
    """

    original_affine: np.ndarray
    original_shape: tuple[int, ...]
    canonical_affine: np.ndarray
    zooms: tuple[float, float, float]


def load_canonical(path: str | Path) -> tuple[np.ndarray, CanonicalGeometry]:
    """
    Load a NIfTI volume and reorient it to canonical RAS.

    Reorientation is a permutation of the axes plus sign flips. It resamples nothing and is
    exactly invertible, which is why it is safe to bake into a frozen cache: the predicted
    mask can be returned to the input grid bit-for-bit.

    nnU-Net's default reader does NOT reorient. Without this step the network would see the
    anatomy transposed or mirrored from one center to the next, which is precisely the
    heterogeneity this challenge is built around.

    Inputs:
        path: path to a NIfTI file
    Outputs:
        array: float32 volume in RAS, shape (X, Y, Z)
        geom:  CanonicalGeometry carrying the inverse transform
    """
    img = nib.load(str(path))
    original_affine = np.asarray(img.affine, dtype=np.float64)
    original_shape = tuple(int(s) for s in img.shape[:3])

    canonical = nib.as_closest_canonical(img)
    # np.asanyarray(dataobj) reads in the on-disk dtype. get_fdata() would upcast the whole
    # volume to float64 first, doubling peak memory for no gain.
    array = np.asanyarray(canonical.dataobj).astype(np.float32, copy=False)

    geom = CanonicalGeometry(
        original_affine=original_affine,
        original_shape=original_shape,
        canonical_affine=np.asarray(canonical.affine, dtype=np.float64),
        zooms=tuple(float(z) for z in canonical.header.get_zooms()[:3]),
    )
    return array, geom


def restore_native(array: np.ndarray, geom: CanonicalGeometry) -> nib.Nifti1Image:
    """
    Map a canonical (RAS) array back onto the original voxel grid.

    This is the exact inverse of load_canonical. It is asserted, not assumed: a silent
    orientation error would produce a mirrored mask whose Dice collapses without any
    exception being raised.

    Inputs:
        array: volume defined on the canonical grid (e.g. a predicted lesion mask)
        geom:  geometry returned by load_canonical for the SAME case
    Outputs:
        nib.Nifti1Image on the original grid, with the original affine and shape
    """
    canonical_img = nib.Nifti1Image(array, geom.canonical_affine)

    # ornt_transform(start, end) returns the orientation transform taking an array laid out
    # in `start` to the layout of `end`, both expressed relative to RAS. Here start is the
    # canonical (identity) orientation and end is the orientation of the input file.
    back = ornt_transform(
        io_orientation(geom.canonical_affine),
        io_orientation(geom.original_affine),
    )
    native_img = canonical_img.as_reoriented(back)

    assert tuple(native_img.shape[:3]) == geom.original_shape, (
        f"restore_native produced shape {native_img.shape[:3]}, "
        f"expected {geom.original_shape}"
    )
    assert np.allclose(native_img.affine, geom.original_affine, atol=1e-4), (
        "restore_native produced an affine that differs from the input affine"
    )
    return native_img


# --------------------------------------------------------------------------------------
# Intensity
# --------------------------------------------------------------------------------------
def brain_mask(array: np.ndarray) -> np.ndarray:
    """
    Derive the brain mask from a skull-stripped volume.

    SynthStrip sets the background to exactly zero, so strict positivity is the mask. No
    thresholding heuristic is applied: any such heuristic would behave differently across
    centers, reintroducing the site effect the pipeline is trying to remove.

    Inputs:
        array: canonical volume
    Outputs:
        boolean mask of the brain
    """
    return array > 0


def n4_bias_correction(
    array: np.ndarray,
    zooms: tuple[float, float, float],
    shrink_factor: int = 4,
    iterations: tuple[int, ...] = (50, 50, 50, 50),
) -> np.ndarray:
    """
    Apply N4ITK bias field correction inside the brain mask.

    The bias field is estimated on a shrunk volume (a low-frequency field does not need full
    resolution) and then evaluated at full resolution, which is the standard cost/accuracy
    compromise and keeps the step around 30 s per case.

    The SimpleITK image is built with the canonical spacing and an identity direction. N4
    parameterizes its B-spline grid in physical space, so spacing is what matters; the
    direction cosines are irrelevant to the estimated field and omitting them keeps this
    function a pure array-to-array transform, hence trivially reproducible in the container.

    Background voxels are forced back to exactly zero: the division by the bias field is
    undefined outside the mask, and a nonzero background would break the nonzero-crop and
    the masked z-score performed downstream by nnU-Net.

    Inputs:
        array:         canonical float32 volume
        zooms:         voxel spacing (mm), x/y/z, of the canonical volume
        shrink_factor: downsampling factor used for field estimation
        iterations:    maximum N4 iterations per resolution level
    Outputs:
        corrected float32 volume, same shape, background exactly zero
    """
    import SimpleITK as sitk

    mask_np = brain_mask(array)
    if not mask_np.any():
        # An all-zero volume is a data error, surfaced by the QA pass. Correcting it is
        # meaningless, so it is returned untouched rather than crashing the whole batch.
        return array.astype(np.float32, copy=True)

    # SimpleITK indexes arrays as (z, y, x) while nibabel yields (x, y, z).
    img = sitk.GetImageFromArray(np.transpose(array, (2, 1, 0)))
    img.SetSpacing(tuple(float(z) for z in zooms))
    msk = sitk.GetImageFromArray(np.transpose(mask_np.astype(np.uint8), (2, 1, 0)))
    msk.SetSpacing(tuple(float(z) for z in zooms))

    img = sitk.Cast(img, sitk.sitkFloat32)
    img_small = sitk.Shrink(img, [shrink_factor] * 3)
    msk_small = sitk.Shrink(msk, [shrink_factor] * 3)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(iterations))
    corrector.Execute(img_small, msk_small)

    # The field is re-evaluated at full resolution from the B-spline coefficients fitted on
    # the shrunk volume, so no upsampling artefact is introduced in the correction.
    log_bias = corrector.GetLogBiasFieldAsImage(img)
    corrected = img / sitk.Exp(log_bias)

    out = np.transpose(sitk.GetArrayFromImage(corrected), (2, 1, 0)).astype(np.float32)
    out[~mask_np] = 0.0
    # N4 can produce small negatives where the field is poorly conditioned at the brain edge.
    # They are clipped: a negative T1 intensity has no meaning and would leak outside the
    # nonzero mask used for cropping and normalization.
    np.clip(out, 0.0, None, out=out)
    return out


# --------------------------------------------------------------------------------------
# Public contract
# --------------------------------------------------------------------------------------
def preprocess_image(
    path: str | Path,
    apply_n4: bool = False,
) -> tuple[np.ndarray, CanonicalGeometry]:
    """
    Full canonical preprocessing of one T1w volume. THIS IS THE CONTRACT.

    The offline cache builder and the Docker inference entrypoint must both call this
    function and nothing else. Any transformation applied on one side only is a train/test
    skew, and on a challenge scored on hidden centers it is undetectable locally.

    Inputs:
        path:     path to the raw T1w NIfTI (native space, skull-stripped)
        apply_n4: whether to run N4ITK bias correction (cache variant selector)
    Outputs:
        array: preprocessed float32 canonical volume
        geom:  CanonicalGeometry, required to write the prediction back in native space
    """
    array, geom = load_canonical(path)

    # Non-finite voxels occur in a small number of clinical exports. They are neutralized
    # here rather than in the QA pass because N4 and the downstream statistics would both
    # propagate them across the whole volume.
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    if apply_n4:
        array = n4_bias_correction(array, geom.zooms)

    return array.astype(np.float32, copy=False), geom


def binarize_mask(raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Convert a possibly float-valued lesion mask to a strict {0, 1} uint8 mask.

    Some annotation exports store the mask as float with values such as 0.00, 0.99, 1.00, or
    intermediate fractions introduced by an upstream resampling. The rule is nearest-integer
    rounding, then any positive label collapses to lesion:
        0.99 -> 1, 1.00 -> 1, 0.40 -> 0, 0.00 -> 0, 2.0 -> 1.
    Reorientation to RAS is a permutation and introduces no new fractional values, so the raw
    values inspected here are exactly the values stored on disk.

    The diagnostics are computed on the RAW values, before rounding, so that a float or
    multi-label mask is surfaced in the QA report rather than silently normalized away.

    Inputs:
        raw: mask array as read from disk (float or int), any dtype
    Outputs:
        binary: uint8 array in {0, 1}
        diag:   dict describing the raw values (for the QA report)
    """
    finite = raw[np.isfinite(raw)]
    off_integer = (
        np.abs(finite - np.rint(finite)) > 1e-6 if finite.size else np.array([], dtype=bool)
    )
    diag = {
        "mask_raw_min": float(finite.min()) if finite.size else 0.0,
        "mask_raw_max": float(finite.max()) if finite.size else 0.0,
        "mask_raw_is_integer": bool(finite.size and not off_integer.any()),
        "mask_off_integer_fraction": float(off_integer.mean()) if finite.size else 0.0,
        "mask_multilabel": bool(finite.size and finite.max() > 1.5),
    }

    # Nearest-integer rounding, then threshold at > 0. np.rint uses round-half-to-even, which
    # only matters at an exact 0.5 (rounded down to 0); no real annotation sits there.
    binary = (np.rint(raw) > 0).astype(np.uint8)
    return binary, diag


def load_mask_canonical(
    path: str | Path,
    reference: CanonicalGeometry,
    atol: float = 1e-3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load a lesion mask, reorient it to RAS, verify its grid, and binarize it.

    RAS reorientation and grid verification protect against a mask stored in a different
    orientation or on a shifted grid than its image, which would silently destroy every
    voxel-wise metric. Binarization is delegated to binarize_mask (nearest-integer rounding).
    No small-object removal and no hole filling: lesion count and empty-GT cases are ranked
    metrics, so any cleaning of the target changes the objective the leaderboard measures.

    Inputs:
        path:      path to the NIfTI lesion mask
        reference: geometry of the corresponding T1w image
        atol:      tolerance on the affine comparison
    Outputs:
        mask: uint8 mask on the canonical grid, values in {0, 1}
        diag: raw-value diagnostics from binarize_mask, for the QA report
    Raises:
        ValueError: when the mask and the image do not share a voxel grid, which would make
                    every downstream voxel-wise statistic meaningless
    """
    raw, geom = load_canonical(path)

    if geom.original_shape != reference.original_shape:
        raise ValueError(
            f"mask shape {geom.original_shape} != image shape {reference.original_shape}"
        )
    if not np.allclose(geom.original_affine, reference.original_affine, atol=atol):
        raise ValueError("mask affine differs from image affine beyond tolerance")

    return binarize_mask(raw)


# --------------------------------------------------------------------------------------
# Quality assurance
# --------------------------------------------------------------------------------------
def qc_case(
    image: np.ndarray,
    geom: CanonicalGeometry,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Collect per-case QA indicators. Reports; never repairs.

    The published external validation of nnU-Net on ATLAS-family cohorts attributes part of
    the residual Dice error to annotation defects, including voxels labelled outside the
    brain. Those cases must be inventoried before training, not discovered after it, but they
    must not be silently edited either: the ground truth is the target the leaderboard uses.

    Inputs:
        image: canonical volume
        geom:  its geometry
        mask:  canonical lesion mask, or None for test-time cases
    Outputs:
        flat dict of indicators, one row of the QA report
    """
    brain = brain_mask(image)
    n_vox = int(image.size)
    background_fraction = float(1.0 - brain.sum() / max(n_vox, 1))

    rec: dict[str, Any] = {
        "shape": "x".join(str(s) for s in geom.original_shape),
        "zooms": "x".join(f"{z:.3f}" for z in geom.zooms),
        "orientation": "".join(nib.aff2axcodes(geom.original_affine)),
        "is_anisotropic": bool(max(geom.zooms) / max(min(geom.zooms), 1e-6) > 1.5),
        "background_fraction": round(background_fraction, 4),
        "skullstrip_suspect": bool(background_fraction < MIN_BACKGROUND_FRACTION),
        "has_nonfinite": bool(not np.isfinite(image).all()),
        "intensity_p99": float(np.percentile(image[brain], 99)) if brain.any() else 0.0,
        "brain_empty": bool(not brain.any()),
    }

    if mask is None:
        return rec

    lesion = mask > 0
    n_lesion = int(lesion.sum())
    # Lesion voxels falling on a zero-intensity voxel: either an annotation defect or a
    # skull-strip that removed labelled tissue. Either way the model is being asked to
    # predict a lesion where it has no signal at all.
    outside = int(np.logical_and(lesion, ~brain).sum())

    # Note: raw-value integrity of the mask (float labels such as 0.99, multi-label) is NOT
    # checked here. qc_case receives the already-binarized mask, so such a check would be dead
    # code. Those diagnostics are produced by binarize_mask and merged into the QA row by the
    # cache builder instead.
    rec.update(
        {
            "n_lesion_voxels": n_lesion,
            "mask_empty": bool(n_lesion == 0),
            "lesion_voxels_outside_brain": outside,
            "pct_lesion_outside_brain": (
                round(100.0 * outside / n_lesion, 2) if n_lesion else 0.0
            ),
        }
    )
    return rec


# --------------------------------------------------------------------------------------
# Metadata conditioning
# --------------------------------------------------------------------------------------
def fit_metadata_encoding(days: np.ndarray) -> dict[str, float]:
    """
    Fit the normalization constants of the DAYS_POST_STROKE covariate.

    MUST be fitted on the training folds only. Fitting on the full cohort, holdout included,
    leaks the holdout distribution into the model input and voids it as an estimate of the
    hidden test.

    log1p is applied first: days post stroke spans roughly 0 to 10^4 and is strongly skewed,
    so a raw z-score would compress the entire acute range, which is exactly the regime the
    conditioning is meant to disambiguate.

    Robust statistics (median / IQR) are used because the tail is heavy.

    Inputs:
        days: 1D array of DAYS_POST_STROKE over the training folds, NaN allowed
    Outputs:
        dict with the constants to be frozen and shipped inside the container
    """
    d = np.log1p(np.asarray(days, dtype=float))
    d = d[np.isfinite(d)]
    q25, q50, q75 = np.percentile(d, [25, 50, 75])
    iqr = float(q75 - q25) or 1.0
    return {"log1p_median": float(q50), "log1p_iqr": iqr}


def encode_metadata(
    days_post_stroke: float | None,
    chronicity: float | None,
    encoding: dict[str, float],
) -> dict[str, float]:
    """
    Encode the per-case covariates into the FiLM conditioning vector.

    Design decisions:
        - CENTER is NOT encoded. The hidden test set is dominated by centers absent from
          training, so a center embedding resolves to an unknown token at inference and
          contributes nothing there, while offering the network a site shortcut during
          training. CENTER is used for the split and for domain randomization, not as a
          network input.
        - Every covariate is paired with an explicit *_missing flag. DAYS_POST_STROKE is NA
          for a substantial share of the cohort, and the challenge exposes the same field at
          test time, so absence is a state the network must handle, not an edge case.
        - The training loop is expected to drop the conditioning (set both *_missing flags to
          1) for a fraction of samples, so that the model degrades gracefully rather than
          becoming dependent on a covariate that may be absent at test time.

    Inputs:
        days_post_stroke: days between onset and acquisition, or None / NaN
        chronicity:       1 if < 180 days post stroke, 0 otherwise, or None
        encoding:         constants returned by fit_metadata_encoding
    Outputs:
        dict with keys days_norm, days_missing, chronicity, chronicity_missing
    """
    days_missing = days_post_stroke is None or not np.isfinite(float(days_post_stroke))
    if days_missing:
        days_norm = MISSING_SENTINEL
    else:
        days_norm = (
            np.log1p(float(days_post_stroke)) - encoding["log1p_median"]
        ) / encoding["log1p_iqr"]

    chron_missing = chronicity is None or not np.isfinite(float(chronicity))
    chron_val = MISSING_SENTINEL if chron_missing else float(chronicity)

    return {
        "days_norm": float(days_norm),
        "days_missing": float(days_missing),
        "chronicity": chron_val,
        "chronicity_missing": float(chron_missing),
    }


def read_case_metadata(json_path: str | Path) -> dict[str, Any]:
    """
    Read the challenge-provided per-case metadata JSON.

    Key lookup is case-insensitive because the field naming is documented on the challenge
    page but not contractually guaranteed across batches. A missing key yields None, which
    encode_metadata turns into an explicit missing flag rather than a crash.

    Inputs:
        json_path: path to the case .json
    Outputs:
        dict with DAYS_POST_STROKE, CHRONICITY, CENTER (None when absent)
    """
    with open(json_path) as f:
        raw = json.load(f)
    lower = {str(k).lower(): v for k, v in raw.items()}

    def _num(key: str) -> float | None:
        v = lower.get(key)
        if v is None or (isinstance(v, str) and v.strip().upper() in {"NA", "", "NAN"}):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "DAYS_POST_STROKE": _num("days_post_stroke"),
        "CHRONICITY": _num("chronicity"),
        "CENTER": lower.get("center"),
    }