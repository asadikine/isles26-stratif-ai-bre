"""
ISLES'26 - Grand Challenge inference entry point (Path A, 3-fold ensemble).

PURPOSE
    Reproduce, inside the submission container and case by case, the exact inference path
    validated offline by benchmark_holdout.py. Any divergence between the two is a train/test
    skew that is undetectable on the hidden test set, so this module deliberately reuses the
    same shared contract (isles26_preprocess) rather than reimplementing preprocessing.

ENSEMBLING
    The ensemble is not reimplemented here. nnUNetPredictor averages the softmax outputs of
    the folds passed to use_folds before the argmax, patch by patch, in the resampled space.
    This is the same probability averaging measured by the benchmark under the name
    "ensemble". Averaging binary masks after the fact would be a different, weaker operator
    and would destroy the calibration that the challenge PR-AUC metric measures.

PIPELINE / DATA FLOW
    [startup, once per container, driven by app.py]
        init_model()
            -> nnUNetPredictor instantiated
            -> folds ENSEMBLE_FOLDS restored from /opt/ml/model
            -> predictor held in memory for every subsequent case

    [per case, once per POST /invoke]
        run(model) -> interf0_handler(model)

            /input/images/<t1 slug>/case.mha | case.nii.gz      native geometry
                |
                v
            [1] Convert to NIfTI in /tmp when the input is .mha, so that the nibabel-based
                shared contract can read it. SimpleITK writes a valid NIfTI affine, so the
                round trip is geometry-preserving.
                |
                v
            [2] preprocess_image(apply_n4=False)  ->  RAS array + CanonicalGeometry.
                Same call as build_training_cache.py: the network sees the grid it was
                trained on. The geometry carries the exact inverse transform.
                |
                v
            [3] Write the RAS volume to /tmp as .nii.gz and read it back with SimpleITKIO.
                This detour is intentional: nnU-Net planned and preprocessed the training
                cache through SimpleITKIO, whose array order and spacing convention are the
                reverse of nibabel's. Passing a nibabel-ordered array directly to the
                predictor would silently resample along the wrong axes on anisotropic data.
                |
                v
            [4] predict_single_npy_array with use_folds=ENSEMBLE_FOLDS.
                Returns the argmax segmentation and the averaged softmax volume, both on the
                RAS grid, in SimpleITKIO axis order.
                |
                v
            [5] Transpose back to nibabel order, derive the binary mask from the lesion
                probability at a frozen threshold, remove connected components below a
                frozen size.
                |
                v
            [6] restore_native() maps both volumes onto the original voxel grid. This is an
                exact axis permutation with no interpolation; the function asserts that the
                recovered shape and affine match the input.
                |
                v
            /output/images/stroke-lesion-segmentation/output.mha   uint8   {0, 1}
            /output/images/lesion-probability-map/output.mha       float32 [0, 1]

RUNTIME CONTRACT
    - No network access. All weights are read from the model tarball mounted read-only at
      /opt/ml/model.
    - The only writable paths are /output and /tmp.
    - Model loading belongs to init_model(): /health must not return 200 before the three
      folds are resident, and the per-case timeout does not budget for it.

PREREQUISITES INSIDE THE IMAGE
    - nnunetv2, SimpleITK, nibabel, numpy, scipy, scikit-image, torch
    - isles26_preprocess.py on the PYTHONPATH
    - the custom trainer class registered inside the nnunetv2 package (see Dockerfile)
    - the custom normalization class registered inside the nnunetv2 package, when the plans
      reference it
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from skimage.morphology import remove_small_objects

from isles26_preprocess import preprocess_image, restore_native

# --------------------------------------------------------------------------------------
# Platform paths
# --------------------------------------------------------------------------------------
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

# Grand Challenge extracts the uploaded model tarball here. The local ./model directory is
# bind-mounted to the same location during the rehearsal, so this constant never changes.
MODEL_PATH = Path("/opt/ml/model")

# Root of the nnU-Net artefacts inside the model tarball. The dataset and trainer
# configuration folder names are NOT hard-coded: they encode the dataset id, the trainer
# variant and the plans identifier, all of which change between experiments. Hard-coding them
# makes the container silently incompatible with a retrained model, a failure that only
# surfaces as a health check that never succeeds.
NNUNET_ROOT = MODEL_PATH / "nnunet"


def discover_model_folder(root: Path = NNUNET_ROOT) -> Path:
    """
    Locate the nnU-Net trainer configuration folder inside the model tarball.

    The folder is identified by the presence of plans.json and dataset.json, the two files the
    predictor reads at restore time. Searching for them rather than assuming a path makes the
    container tolerant to a change of dataset id, trainer variant or plans identifier.

    Inputs:
        root: directory holding the extracted nnU-Net artefacts
    Outputs:
        Path to the trainer configuration folder.
    Raises:
        FileNotFoundError: when no candidate is found, listing what is present so the
                           mismatch can be diagnosed from the container logs alone
        RuntimeError:      when several candidates exist, since the choice would be arbitrary
    """
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} does not exist. The model tarball is missing or was packed from the "
            f"wrong level. Contents of {MODEL_PATH}: "
            f"{sorted(p.name for p in MODEL_PATH.glob('*')) if MODEL_PATH.is_dir() else 'absent'}"
        )

    candidates = sorted(
        plans.parent
        for plans in root.rglob("plans.json")
        if (plans.parent / "dataset.json").exists()
    )

    if not candidates:
        listing = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        raise FileNotFoundError(
            f"no folder containing both plans.json and dataset.json under {root}. "
            f"Files present: {listing[:20]}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"several model folders found under {root}: {[str(c) for c in candidates]}. "
            f"The tarball must contain exactly one trained configuration."
        )
    return candidates[0]

# Output socket slugs, fixed by the challenge interface definition.
SEGMENTATION_SLUG = "stroke-lesion-segmentation"
PROBABILITY_SLUG = "lesion-probability-map"

# --------------------------------------------------------------------------------------
# Frozen inference parameters
# --------------------------------------------------------------------------------------
# Folds participating in the ensemble. Softmax averaging across them is performed inside the
# predictor. Changing this tuple changes the model, not a runtime setting.
ENSEMBLE_FOLDS = (0, 1, 2)
CHECKPOINT_NAME = "checkpoint_final.pth"

# Decision threshold selected on the holdout by benchmark_holdout.py and frozen here. It is a
# model artefact: it is never re-derived from the test data.
BINARY_THRESHOLD = 0.5

# Minimum connected-component size, in voxels, applied on the RAS grid before reprojection.
# The value MUST reproduce whatever postprocessing the offline benchmark actually scored,
# otherwise the holdout figures no longer describe this container.
#
# It defaults to 0, meaning no filtering, because benchmark_holdout.py falls back to scoring
# raw predictions whenever postprocessing.pkl is absent, which is the case unless
# nnUNetv2_find_best_configuration or nnUNetv2_determine_postprocessing was run. Check the
# benchmark log: if it printed "no postprocessing.pkl found; scoring raw predictions", leave
# this at 0.
#
# Raising it is a modelling decision, not a safety net: the ranking is driven by Detection F1
# and lesion-count difference, so discarding small multifocal lesions is penalised twice.
# Measure any nonzero value on the holdout before adopting it.
MIN_COMPONENT_SIZE = 0

# Sliding-window step and test-time mirroring. Mirroring performs eight forward passes per
# case and per fold, i.e. twenty-four passes for the ensemble. If the per-case runtime budget
# is exceeded on the platform, disable mirroring here rather than dropping a fold: the
# benchmark quantified the cost of each option separately.
TILE_STEP_SIZE = 0.5
USE_MIRRORING = True


# --------------------------------------------------------------------------------------
# Model lifecycle
# --------------------------------------------------------------------------------------
def init_model() -> nnUNetPredictor:
    """
    Build the nnU-Net predictor and restore the ensemble weights.

    Called once by app.py during the server lifespan, before /health returns 200. The three
    checkpoints are resident from that point on and are reused for every case.

    Restoring the checkpoints requires the trainer class named inside them
    (nnUNetTrainerISLES26A) to be importable from the nnunetv2 package tree, and the
    normalization scheme named in plans.json to be resolvable from
    nnunetv2.preprocessing.normalization. Both are installed by the Dockerfile.

    Outputs:
        nnUNetPredictor with folds ENSEMBLE_FOLDS restored and ready to predict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"inference device: {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_properties(0)}", flush=True)

    model_folder = discover_model_folder()
    print(f"model folder: {model_folder}", flush=True)

    predictor = nnUNetPredictor(
        tile_step_size=TILE_STEP_SIZE,
        use_gaussian=True,
        use_mirroring=USE_MIRRORING,
        # Keeping resampling and window aggregation on the GPU is faster but raises peak
        # memory. The platform allocates 16 GB; a fallback is applied per case if it is
        # exhausted, see predict_ensemble.
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=ENSEMBLE_FOLDS,
        checkpoint_name=CHECKPOINT_NAME,
    )
    print(f"loaded nnU-Net ensemble, folds {ENSEMBLE_FOLDS}", flush=True)

    # The postprocessing chain is attached to the predictor so that it is read once at
    # startup, like the weights, rather than on every case.
    predictor.isles26_postprocessing = load_postprocessing(model_folder)
    return predictor


def run(model: nnUNetPredictor) -> int:
    """
    Dispatch one case to the handler matching the active interface.

    Inputs:
        model: predictor returned by init_model()
    Outputs:
        0 on success. app.py turns this into HTTP 201.
    """
    interface_key = get_interface_key()
    print(f"active interface: {interface_key}", flush=True)

    # A single interface is defined for this task. The lookup is kept explicit so that an
    # unexpected socket combination fails loudly here rather than producing a silent
    # mis-selection deeper in the pipeline.
    handlers = {
        ("stroke-metadata", "t1-brain-mri"): interf0_handler,
    }
    handler = handlers.get(interface_key, interf0_handler)
    return handler(model)


def interf0_handler(model: nnUNetPredictor) -> int:
    """
    Process one T1-weighted case and write both required outputs.

    Inputs:
        model: initialized nnUNetPredictor
    Outputs:
        0 on success.
    """
    image_path = resolve_input_image()
    print(f"input image: {image_path}", flush=True)

    metadata = read_optional_metadata()
    print(f"case metadata: {json.dumps(metadata)}", flush=True)

    with tempfile.TemporaryDirectory(dir="/tmp") as scratch:
        scratch_dir = Path(scratch)

        # The shared contract reads NIfTI through nibabel. Grand Challenge may deliver .mha,
        # so the volume is transcoded first. SimpleITK writes a standards-compliant NIfTI
        # affine, making the round trip geometry-preserving.
        nifti_path = ensure_nifti(image_path, scratch_dir)

        # Same call as the offline cache builder. apply_n4 must match the variant the model
        # was trained on.
        ras_array, geometry = preprocess_image(nifti_path, apply_n4=False)

        probability_ras, segmentation_ras = predict_ensemble(model, ras_array, geometry, scratch_dir)

    binary_ras = binarize(probability_ras, fallback=segmentation_ras)

    # Reproduce the offline benchmark: the same postprocessing chain, applied on the same
    # grid, before the return to native space.
    functions, kwargs = getattr(model, "isles26_postprocessing", (None, None))
    binary_ras = apply_postprocessing(binary_ras, functions, kwargs)

    # Reprojection to the native grid: an exact axis permutation. restore_native asserts that
    # the recovered shape and affine match the input, so an orientation error raises here
    # instead of producing a mirrored mask with a silently collapsed Dice.
    binary_native = restore_native(binary_ras.astype(np.uint8), geometry)
    probability_native = restore_native(probability_ras.astype(np.float32), geometry)

    reference = sitk.ReadImage(str(image_path))
    write_prediction(OUTPUT_PATH / "images" / SEGMENTATION_SLUG, binary_native, reference, np.uint8)
    write_prediction(OUTPUT_PATH / "images" / PROBABILITY_SLUG, probability_native, reference, np.float32)

    print(
        f"predicted lesion volume: {float(np.asanyarray(binary_native.dataobj).sum()):.0f} voxels",
        flush=True,
    )
    return 0


# --------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------
def predict_ensemble(
    predictor: nnUNetPredictor,
    ras_array: np.ndarray,
    geometry,
    scratch_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the fold ensemble on a canonical RAS volume.

    The volume is written to a temporary NIfTI and read back with SimpleITKIO rather than
    handed to the predictor as a raw array. nnU-Net planned the training cache through
    SimpleITKIO, which stores the array as (z, y, x) and the spacing in the matching reversed
    order, whereas nibabel yields (x, y, z). Feeding the nibabel-ordered array directly would
    make the predictor resample along the wrong axes, a failure that is invisible on isotropic
    data and severe on anisotropic acquisitions.

    Inputs:
        predictor  : initialized nnUNetPredictor
        ras_array  : canonical float32 volume, nibabel axis order (x, y, z)
        geometry   : CanonicalGeometry for this case, supplies the canonical affine and zooms
        scratch_dir: writable temporary directory under /tmp
    Outputs:
        probability: float32 lesion probability on the RAS grid, nibabel axis order
        segmentation: uint8 nnU-Net argmax on the RAS grid, nibabel axis order
    """
    staged = scratch_dir / "case_0000.nii.gz"
    image_nii = nib.Nifti1Image(ras_array.astype(np.float32), geometry.canonical_affine)
    image_nii.header.set_zooms(geometry.zooms)
    nib.save(image_nii, str(staged))

    image, properties = SimpleITKIO().read_images([str(staged)])

    try:
        segmentation, probabilities = predictor.predict_single_npy_array(
            input_image=image,
            image_properties=properties,
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=True,
        )
    except torch.cuda.OutOfMemoryError:
        # Fallback for the largest volumes on a 16 GB device: move resampling and window
        # aggregation to host memory. Slower, but it produces the same result rather than
        # failing the case.
        print("cuda oom, retrying with host-side aggregation", flush=True)
        torch.cuda.empty_cache()
        predictor.perform_everything_on_device = False
        segmentation, probabilities = predictor.predict_single_npy_array(
            input_image=image,
            image_properties=properties,
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=True,
        )
        predictor.perform_everything_on_device = True

    # Channel 1 is the lesion class in a binary configuration.
    probability = np.ascontiguousarray(probabilities[1]).astype(np.float32)

    # Back to nibabel axis order, so both volumes live on the same grid as ras_array and can
    # be passed to restore_native.
    probability = np.transpose(probability, (2, 1, 0))
    segmentation = np.transpose(np.asarray(segmentation), (2, 1, 0)).astype(np.uint8)

    if probability.shape != ras_array.shape:
        raise RuntimeError(
            f"probability shape {probability.shape} does not match the canonical volume "
            f"{ras_array.shape}; the axis convention assumption is violated"
        )
    return probability, segmentation


def load_postprocessing(model_folder: Path):
    """
    Load the nnU-Net postprocessing pipeline determined by find_best_configuration.

    nnUNetv2_find_best_configuration writes postprocessing.pkl under
    crossval_results_folds_<f>/ inside the trainer configuration folder. The pickle holds a
    pair (functions, keyword arguments): the functions are pickled by reference, so they
    unpickle to the real callables provided nnunetv2 is importable, which it is here.

    Applying it is not optional when the offline benchmark applied it: the holdout scores then
    describe post-processed predictions, and a container that skips the step reports different
    numbers on the same data. Conversely, applying it when the benchmark did not is equally
    wrong. Consistency with what was measured is the requirement, not the presence or absence
    of the step.

    Inputs:
        model_folder: trainer configuration folder inside the model tarball
    Outputs:
        (functions, kwargs) when a pickle is present, otherwise (None, None)
    """
    candidates = sorted(model_folder.glob("crossval_results_folds_*/postprocessing.pkl"))
    if not candidates:
        candidates = sorted(model_folder.glob("postprocessing.pkl"))
    if not candidates:
        print("no postprocessing.pkl in the model tarball, raw predictions are written",
              flush=True)
        return None, None

    with open(candidates[0], "rb") as handle:
        functions, kwargs = pickle.load(handle)

    names = [getattr(fn, "__name__", str(fn)) for fn in functions]
    print(f"postprocessing loaded from {candidates[0].name}: {names}", flush=True)
    return functions, kwargs


def apply_postprocessing(segmentation: np.ndarray, functions, kwargs) -> np.ndarray:
    """
    Apply the loaded postprocessing chain to a segmentation on the RAS grid.

    The chain is applied before reprojection, which is where nnU-Net applies it too, so that
    connected components are evaluated on the grid the model predicted on.

    Inputs:
        segmentation: uint8 label map on the RAS grid
        functions   : callables from the postprocessing pickle, or None
        kwargs      : matching keyword argument dicts, or None
    Outputs:
        uint8 label map, post-processed when a chain was provided.
    """
    if not functions:
        return segmentation

    result = segmentation
    for function, function_kwargs in zip(functions, kwargs):
        result = function(result, **function_kwargs)
    return result.astype(np.uint8)


def binarize(probability: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """
    Derive the binary mask from the probability volume.

    The mask is recomputed from the probabilities rather than reusing nnU-Net's argmax, so
    that the operating point remains an explicit, auditable parameter. The argmax is kept as a
    consistency reference only.

    Inputs:
        probability: float32 volume in [0, 1] on the RAS grid
        fallback   : nnU-Net argmax on the same grid, used for a sanity comparison
    Outputs:
        uint8 array with values in {0, 1}
    """
    mask = probability > BINARY_THRESHOLD
    if MIN_COMPONENT_SIZE > 1:
        mask = remove_small_objects(mask, min_size=MIN_COMPONENT_SIZE)

    thresholded = int(mask.sum())
    argmaxed = int((fallback > 0).sum())
    print(f"foreground voxels: threshold={thresholded}, nnunet_argmax={argmaxed}", flush=True)
    return mask.astype(np.uint8)


# --------------------------------------------------------------------------------------
# Input and output handling
# --------------------------------------------------------------------------------------
def resolve_input_image() -> Path:
    """
    Locate the T1-weighted volume provided for the current case.

    The image socket directory is discovered rather than hard-coded. The challenge template
    documents the slug as t1-brain-mri while the offline benchmark used
    skull-stripped-t1-brain-mri; scanning /input/images removes the risk of a mismatch between
    the container and the interface actually configured on the algorithm page.

    Outputs:
        Path to the image file.
    """
    patterns = ("*.mha", "*.nii.gz", "*.nii", "*.mhd")
    for socket_dir in sorted((INPUT_PATH / "images").glob("*")):
        if not socket_dir.is_dir():
            continue
        candidates: list[str] = []
        for pattern in patterns:
            candidates.extend(glob.glob(str(socket_dir / pattern)))
        if candidates:
            return Path(sorted(candidates)[0])
    raise FileNotFoundError(f"no image file found under {INPUT_PATH / 'images'}")


def ensure_nifti(image_path: Path, scratch_dir: Path) -> Path:
    """
    Return a NIfTI path for the input volume, transcoding when required.

    Inputs:
        image_path : path to the case volume, any format SimpleITK can read
        scratch_dir: writable temporary directory
    Outputs:
        Path to a NIfTI file readable by nibabel.
    """
    if image_path.name.endswith((".nii", ".nii.gz")):
        return image_path
    converted = scratch_dir / "input.nii.gz"
    sitk.WriteImage(sitk.ReadImage(str(image_path)), str(converted), useCompression=True)
    return converted


def read_optional_metadata() -> dict:
    """
    Read the per-case metadata JSON when present.

    Path A does not condition on metadata, so absence is not an error. The values are logged
    for traceability of the test-phase run. Note that the acquisition site key is named CENTER
    on the platform and SITE in the training CSV, and that DAYS_POST_STROKE and CHRONICITY may
    be null.

    Outputs:
        Parsed metadata dict, empty when the file is absent.
    """
    location = INPUT_PATH / "stroke-metadata.json"
    if not location.exists():
        return {}
    with open(location) as handle:
        return json.loads(handle.read())


def get_interface_key() -> tuple[str, ...]:
    """
    Derive the active interface from the platform-generated inputs.json.

    Outputs:
        Sorted tuple of input socket slugs, empty when inputs.json is absent.
    """
    location = INPUT_PATH / "inputs.json"
    if not location.exists():
        return ()
    with open(location) as handle:
        inputs = json.loads(handle.read())
    return tuple(sorted(entry["socket"]["slug"] for entry in inputs))


def write_prediction(location: Path, image: nib.Nifti1Image, reference: sitk.Image, dtype) -> None:
    """
    Write a native-space prediction as a compressed .mha aligned with the input.

    The array is transposed from nibabel order (x, y, z) to SimpleITK order (z, y, x) and the
    spatial metadata is copied from the input image. Copying is what guarantees alignment with
    the reference annotation during evaluation: an image written without it sits at the
    identity transform and scores near zero regardless of the prediction quality.

    Inputs:
        location : output directory, created if absent
        image    : prediction on the native grid
        reference: the input image, supplying spacing, origin and direction
        dtype    : numpy dtype for the written volume
    Outputs:
        None.
    """
    location.mkdir(parents=True, exist_ok=True)
    array = np.transpose(np.asanyarray(image.dataobj), (2, 1, 0)).astype(dtype)

    output = sitk.GetImageFromArray(array)
    if output.GetSize() != reference.GetSize():
        raise RuntimeError(
            f"prediction size {output.GetSize()} does not match input size {reference.GetSize()}"
        )
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(location / "output.mha"), useCompression=True)


if __name__ == "__main__":
    # Direct execution is provided for debugging only. On Grand Challenge the container is
    # driven by app.py through the /health and /invoke endpoints.
    os.environ.setdefault("nnUNet_results", "/tmp/nnUNet_results")
    raise SystemExit(run(model=init_model()))