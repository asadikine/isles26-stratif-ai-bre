"""
ISLES'26 - Model tarball preparation for Grand Challenge.

PURPOSE
    Build the ./model directory that becomes the uploaded model tarball, containing only the
    artefacts the nnU-Net predictor actually reads at inference time. The trained results
    folder holds several gigabytes of training by-products that must not reach the platform.

WHAT IS COPIED
    dataset.json                   channel count, label map, file extension
    plans.json                     architecture, target spacing, patch size, normalization
    fold_<k>/checkpoint_final.pth  network weights, one per ensemble fold, slimmed

WHAT IS DISCARDED
    fold_<k>/validation/           predicted volumes and .npz probabilities, often tens of GB
    fold_<k>/training_log_*.txt, progress.png, debug.json
    checkpoint_best.pth, checkpoint_latest.pth
    crossval_results_folds_*/      unless --keep-postprocessing is passed
    the optimizer and grad scaler states inside each checkpoint

CHECKPOINT SLIMMING
    initialize_from_trained_model_folder reads exactly four keys per checkpoint:
    network_weights, trainer_name, init_args and inference_allowed_mirroring_axes. The
    optimizer and grad scaler states are training artefacts and typically account for two
    thirds of the file size. Removing them is lossless for inference.

PIPELINE / DATA FLOW
    nnUNet_results/<dataset>/<trainer__plans__config>/
        |
        v
    [1] Copy dataset.json and plans.json verbatim.
    [2] For each requested fold, load checkpoint_final.pth on CPU, retain the inference keys,
        write the reduced checkpoint.
    [3] Optionally copy crossval_results_folds_*/postprocessing.pkl.
        |
        v
    model/nnunet/<dataset>/<trainer__plans__config>/

USAGE
    python pack_model.py \
        --results-dir /path/nnUNet_results/Dataset501_ISLES26_raw/nnUNetTrainerISLES26A__nnUNetResEncUNetLPlans__3d_fullres \
        --output-dir ./model \
        --dataset-name Dataset501_ISLES26_raw \
        --folds 0 1 2

    Then, from the repository root:
        tar -czf model.tar.gz -C model .

PREREQUISITES
    torch, and read access to the trained nnUNet_results folder.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

# Keys consumed by nnUNetPredictor.initialize_from_trained_model_folder. The first three are
# mandatory; inference_allowed_mirroring_axes is optional and defaults to None when absent.
INFERENCE_KEYS = (
    "network_weights",
    "trainer_name",
    "init_args",
    "inference_allowed_mirroring_axes",
)
MANDATORY_KEYS = INFERENCE_KEYS[:3]

# Files that must sit at the root of the trainer configuration folder.
PLAN_FILES = ("dataset.json", "plans.json")


def slim_checkpoint(source: Path, destination: Path) -> tuple[int, int]:
    """
    Rewrite one nnU-Net checkpoint keeping only the inference-relevant keys.

    Inputs:
        source     : path to the original checkpoint_final.pth
        destination: path where the reduced checkpoint is written
    Outputs:
        (original_bytes, reduced_bytes) for the size report
    Raises:
        KeyError: when a mandatory key is missing, which indicates the file is not an
                  nnU-Net v2 checkpoint or was produced by an incompatible version
    """
    # weights_only=False is required: the checkpoint stores non-tensor objects (init_args,
    # trainer_name) alongside the state dict.
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)

    missing = set(MANDATORY_KEYS) - set(checkpoint)
    if missing:
        raise KeyError(f"{source}: missing mandatory keys {sorted(missing)}")

    reduced = {key: checkpoint[key] for key in INFERENCE_KEYS if key in checkpoint}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reduced, destination)
    return source.stat().st_size, destination.stat().st_size


def pack(
    results_dir: Path,
    output_dir: Path,
    dataset_name: str,
    folds: list[int],
    keep_postprocessing: bool,
) -> None:
    """
    Assemble the model directory to be archived into the tarball.

    The layout mirrors NNUNET_MODEL_FOLDER in inference.py. Changing one without the other
    produces a container that starts and never becomes healthy.

    Inputs:
        results_dir        : trainer configuration folder inside nnUNet_results
        output_dir         : destination, typically ./model
        dataset_name       : dataset folder name reproduced inside the tarball
        folds              : fold indices to embed in the ensemble
        keep_postprocessing: also copy crossval_results_folds_*/postprocessing.pkl
    Outputs:
        None. Files are written to disk and a size report is printed.
    """
    target = output_dir / "nnunet" / dataset_name / results_dir.name
    target.mkdir(parents=True, exist_ok=True)

    for name in PLAN_FILES:
        source = results_dir / name
        if not source.exists():
            raise FileNotFoundError(f"{source} is required by the predictor and is missing")
        shutil.copy2(source, target / name)
        print(f"copied {name}")

    total_before = 0
    total_after = 0
    for fold in folds:
        source = results_dir / f"fold_{fold}" / "checkpoint_final.pth"
        if not source.exists():
            raise FileNotFoundError(f"{source} not found; fold {fold} was not trained")
        before, after = slim_checkpoint(source, target / f"fold_{fold}" / "checkpoint_final.pth")
        total_before += before
        total_after += after
        print(f"fold {fold}: {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB")

    if keep_postprocessing:
        for candidate in results_dir.glob("crossval_results_folds_*/postprocessing.pkl"):
            destination = target / candidate.parent.name / candidate.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            print(f"copied {candidate.parent.name}/{candidate.name}")

    print(
        f"\nmodel directory ready at {output_dir}\n"
        f"checkpoints: {total_before / 1e6:.0f} MB -> {total_after / 1e6:.0f} MB\n"
        f"next: tar -czf model.tar.gz -C {output_dir} ."
    )


def main() -> int:
    """Parse arguments and build the model directory."""
    parser = argparse.ArgumentParser(description="ISLES'26 model tarball preparation.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="trained model folder, e.g. .../nnUNetTrainerISLES26A__nnUNetResEncUNetLPlans__3d_fullres",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    parser.add_argument("--dataset-name", type=str, default="Dataset501_ISLES26_raw")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--keep-postprocessing",
        action="store_true",
        help="also copy postprocessing.pkl, only useful if inference.py applies it",
    )
    args = parser.parse_args()

    pack(
        args.results_dir,
        args.output_dir,
        args.dataset_name,
        args.folds,
        args.keep_postprocessing,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())