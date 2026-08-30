# ISLES'26 submission image, Path A, 3-fold nnU-Net ensemble.
#
# PIPELINE / DATA FLOW
#   [1] Base image: PyTorch with CUDA 12.6, required by the T4 instances used on the platform.
#   [2] A virtualenv inheriting system site packages, so that pip resolves nnunetv2 without
#       replacing the CUDA-enabled torch of the base image by a CPU wheel.
#   [3] Python dependencies from requirements.txt.
#   [4] Registration of the two project-specific classes inside the installed nnunetv2
#       package. nnU-Net resolves the trainer named in the checkpoint and the normalization
#       named in plans.json by scanning its own package tree, so a file merely present on the
#       PYTHONPATH is NOT found. This step is mandatory and is the most common cause of a
#       container that starts but never becomes healthy.
#   [5] Application code. Model weights are deliberately absent: they are uploaded separately
#       as a model tarball and mounted read-only at /opt/ml/model.

# The registry is stated explicitly. Podman does not assume a default registry for an
# unqualified name: it consults unqualified-search-registries in registries.conf and, when
# several candidates match, prompts interactively. That prompt blocks any non-interactive
# build. Docker Hub is where the official PyTorch images are published.
FROM --platform=linux/amd64 docker.io/pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS isles26_algorithm

# Unbuffered output, so that logs are complete even when the container is terminated.
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

# --system-site-packages keeps the base image torch visible, so pip treats the nnunetv2 torch
# requirement as already satisfied. --without-pip is needed because the base image ships no
# ensurepip; pip itself is inherited from the system site packages.
RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# nnU-Net resolves these three variables at import time. They are irrelevant when predicting
# from an explicit model folder, but must point somewhere writable to avoid import-time
# failures. /tmp is the only writable location besides /output on the platform.
ENV nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/tmp/nnUNet_results

# --- Registration of the project-specific nnU-Net classes -------------------------------
# The checkpoint stores trainer_name="nnUNetTrainerISLES26A". At restore time nnU-Net calls
# recursive_find_python_class over nnunetv2.training.nnUNetTrainer, so the class must live
# inside that package. The same applies to the normalization scheme referenced by plans.json,
# which is resolved from nnunetv2.preprocessing.normalization.
COPY --chown=user:user nnunet_trainer_isles26.py /opt/app/
COPY --chown=user:user nnunet_normalization.py /opt/app/
RUN NNUNET_DIR=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))") && \
    TRAINER_DIR="${NNUNET_DIR}/training/nnUNetTrainer/variants/isles26" && \
    NORM_DIR="${NNUNET_DIR}/preprocessing/normalization" && \
    mkdir -p "${TRAINER_DIR}" && \
    touch "${TRAINER_DIR}/__init__.py" && \
    cp /opt/app/nnunet_trainer_isles26.py "${TRAINER_DIR}/" && \
    cp /opt/app/nnunet_normalization.py "${NORM_DIR}/" && \
    echo "from nnunetv2.preprocessing.normalization.nnunet_normalization import ClippedZScoreNormalization" \
        >> "${NORM_DIR}/__init__.py"

# --- Application code -------------------------------------------------------------------
COPY --chown=user:user isles26_preprocess.py /opt/app/
COPY --chown=user:user app.py /opt/app/
COPY --chown=user:user inference.py /opt/app/

# Fail the build rather than the deployment if a class cannot be resolved.
RUN python -c "\
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class; \
import nnunetv2, os; \
c = recursive_find_python_class(os.path.join(os.path.dirname(nnunetv2.__file__), 'training', 'nnUNetTrainer'), 'nnUNetTrainerISLES26A', 'nnunetv2.training.nnUNetTrainer'); \
assert c is not None, 'trainer nnUNetTrainerISLES26A not resolvable'; \
print('trainer resolved:', c)"

# Required by Grand Challenge to detect that this container implements the invoke API.
# Without it the platform falls back to exec mode and the server is never called.
LABEL org.grand-challenge.api-method="invoke"

EXPOSE 4743
ENTRYPOINT ["python", "app.py"]