#!/usr/bin/env bash
#
# Build the ISLES'26 submission image with Podman.
#
# PIPELINE / DATA FLOW
#   [1] Build for linux/amd64, the architecture of the platform runners.
#   [2] Assert the Grand Challenge API label is present. Without it the platform ignores the
#       HTTP server and the algorithm fails with no useful log.
#   [3] Assert torch is the CUDA build inherited from the base image, not a CPU wheel pulled
#       in by dependency resolution.
#
# Usage: ./do_build_podman.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# Image tag. OCI reference rules require the repository name to be entirely lowercase:
# an uppercase character makes podman reject the build with "invalid reference format".
# Digits, dashes, underscores and periods are allowed.
IMAGE_TAG="${IMAGE_TAG:-isles26-path-a-ensemble}"

echo "> Building ${IMAGE_TAG}"
podman build \
    --platform=linux/amd64 \
    --tag "$IMAGE_TAG" \
    "$SCRIPT_DIR"

echo "> Verifying the Grand Challenge API label"
api_method=$(podman inspect \
    --format '{{index .Config.Labels "org.grand-challenge.api-method"}}' \
    "$IMAGE_TAG")
if [ "$api_method" != "invoke" ]; then
    echo "> ERROR: LABEL org.grand-challenge.api-method=\"invoke\" is missing" >&2
    exit 1
fi
echo "  label: ${api_method}"

echo "> Verifying the torch build"
podman run --rm --entrypoint python "$IMAGE_TAG" -c \
    "import torch; print(f'  torch {torch.__version__}, cuda {torch.version.cuda}'); \
     assert torch.version.cuda is not None, 'CPU-only torch was installed'"

echo "> Verifying that no model weights leaked into the image"
if podman run --rm --entrypoint sh "$IMAGE_TAG" -c "ls /opt/ml/model 2>/dev/null | head -1" | grep -q .; then
    echo "> WARNING: /opt/ml/model is not empty inside the image." >&2
    echo "> Weights must be uploaded as a separate model tarball, not baked in." >&2
fi

echo "> Build complete: ${IMAGE_TAG}"