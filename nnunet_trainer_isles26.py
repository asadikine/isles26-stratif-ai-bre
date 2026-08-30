"""
ISLES'26 - Path A custom nnU-Net v2 trainers.

PURPOSE
    Define the training behaviour for Path A (plain ResEnc-L). The
    architecture itself is NOT set here: it comes from the ResEnc-L plans identifier
    (nnUNetResEncUNetLPlans), selected at plan-and-preprocess time. A trainer in nnU-Net v2 is
    orthogonal to the network topology; it controls sampling, augmentation, schedule and epoch
    count only.

PIPELINE POSITION
    build_training_cache.py  ->  nnUNet_raw / dataset.json / splits_final.json
        |
        v
    nnUNetv2_plan_and_preprocess -pl nnUNetPlannerResEncL   (produces nnUNetResEncUNetLPlans)
        |
        v
    nnUNetv2_train ... -p nnUNetResEncUNetLPlans -tr nnUNetTrainerISLES26A   (THIS FILE)
        |
        v
    checkpoint_final.pth per fold  ->  local inference / Docker

WHAT PATH A CHANGES RELATIVE TO THE STOCK nnUNetTrainer
    1. oversample_foreground_percent 0.33 -> 0.50.
       This is the single highest-yield knob for the ISLES'26 ranking. Raising the fraction 
       of patches centred on a foreground voxel exposes the network to more lesion tissue per 
       epoch. It only affects cases that HAVE a lesion; empty-GT (acute) cases keep drawing
       random patches, which is the desired behaviour.

    2. Nothing else.
       The stock nnU-Net augmentation pack (rotation, scaling, elastic, gamma, brightness,
       contrast, Gaussian noise, Gaussian blur, low-resolution simulation, mirroring) is kept
       verbatim.

WHAT PATH A DELIBERATELY DOES NOT DO
    - No determinism forcing. torch.use_deterministic_algorithms and cudnn.deterministic are
      left untouched, so cuDNN keeps benchmark mode. This is a speed decision: full 3D
      determinism costs 10-20 percent throughput and some 3D ops refuse to run under it.
      Reproducibility is instead anchored by the frozen split and the 5-fold ensemble.
    - No transform-probability tuning in code. Heavier augmentation is available WITHOUT
      editing this file by selecting the built-in nnUNetTrainerDA5 at the command line
      (-tr nnUNetTrainerDA5). Tuning individual transform probabilities requires overriding
      get_training_transforms, whose signature changes across nnU-Net versions; that is
      deferred to the Phase 2 refactor to avoid a version-fragile override here.

INSTALLATION (so nnU-Net can discover the trainer)
    nnU-Net locates trainers by recursively scanning the nnunetv2.training.nnUNetTrainer
    package. Drop this file inside that package tree, e.g.:

        TARGET=$(python -c "import nnunetv2, os; \
            print(os.path.join(os.path.dirname(nnunetv2.__file__), \
            'training','nnUNetTrainer','variants','isles26'))")
        mkdir -p "$TARGET" && touch "$TARGET/__init__.py"
        cp nnunet_trainer_isles26.py "$TARGET/"

    Then reference it by class name with -tr, e.g. -tr nnUNetTrainerISLES26A.

COMPATIBILITY
    Written against the nnUNetTrainer.__init__ signature
        (plans, configuration, fold, dataset_json, device)
    used by nnU-Net v2 (v2.2+). If your installed version differs, align the signature with
    the nnUNetTrainer in your environment rather than guessing.
"""

from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerISLES26A(nnUNetTrainer):
    """
    Path A baseline trainer: ResEnc-L (via plans) with raised foreground oversampling.

    Full-length schedule (inherits num_epochs = 1000 from the base trainer). Use the shorter
    variants below for fast iteration while developing the pipeline.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Inputs:
            plans:         plans dict passed by nnU-Net (ResEnc-L when -p nnUNetResEncUNetLPlans)
            configuration: configuration string, e.g. "3d_fullres"
            fold:          fold index; with the frozen splits_final.json this is a
                           leave-center-out fold, so validation is out-of-domain
            dataset_json:  dataset.json produced by build_training_cache.py
            device:        training device
        """
        super().__init__(plans, configuration, fold, dataset_json, device)

        # 0.33 -> 0.50. Highest-yield change for the lesion-wise ranked metrics. See the
        # module docstring, point 1.
        self.oversample_foreground_percent = 0.50

        # Everything else (loss, lr schedule, augmentation, mirror axes) is inherited
        # unchanged. This is the "measure the baseline before touching anything else"
        # principle made explicit.

class nnUNetTrainerISLES26A_500epochs(nnUNetTrainerISLES26A):
    """
    Intermediate variant: 500 epochs.

    Useful for augmentation A/B comparisons (this trainer vs nnUNetTrainerDA5) where the
    ranking between configurations usually stabilises well before 1000 epochs, at half the
    compute.
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.save_every = 10
