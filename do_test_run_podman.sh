#!/usr/bin/env bash
#
# Local rehearsal of the ISLES'26 container with Podman.
#
# PIPELINE / DATA FLOW
#   [1] Build the image and validate its labels (delegated to do_build_podman.sh).
#   [2] Start the container detached, reproducing the platform mounts:
#         ./model            -> /opt/ml/model  read-only, stands in for the model tarball
#         ./test/input/interf0 -> /input       read-only
#         ./test/output/interf0 -> /output
#   [3] Poll GET /health until the three folds are resident (HTTP 200).
#   [4] Issue POST /invoke and require HTTP 201.
#   [5] Assert both output sockets were written, and report their geometry against the input,
#       which is what catches a reprojection error before submission.
#
# The test case is expected to come from the Grand-Challenge test tree produced by
# benchmark_holdout.py --make-gc-test, so the rehearsal runs on a real holdout volume.
#
# Requires: podman, curl, python3 with SimpleITK for the geometry check.
# Usage: ./do_test_run_podman.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
IMAGE_TAG="${IMAGE_TAG:-isles26-path-a-ensemble}"
CONTAINER_NAME="${IMAGE_TAG}-rehearsal"
PORT=4743

# Case folder to rehearse. Grand Challenge sends one case per /invoke, so several cases are
# tested by running this script once per folder:
#   CASE_DIR=interf0       ./do_test_run_podman.sh
#   CASE_DIR=interf0_case2 ./do_test_run_podman.sh
CASE_DIR="${CASE_DIR:-interf0}"

MODEL_DIR="${SCRIPT_DIR}/model"
INPUT_DIR="${SCRIPT_DIR}/test/input/${CASE_DIR}"
OUTPUT_DIR="${SCRIPT_DIR}/test/output/${CASE_DIR}"

# The health timeout must cover loading three ResEnc-L checkpoints. The platform allows
# roughly five minutes; this mirrors it.
HEALTH_MAX_ATTEMPTS=30
HEALTH_DELAY_SECONDS=10

# A three-fold ensemble with test-time mirroring performs twenty-four forward passes per case.
# On a T4 this is minutes, not seconds.
INVOKE_TIMEOUT_SECONDS=1800

cleanup() {
    podman rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [ ! -d "$MODEL_DIR" ]; then
    echo "> ERROR: ${MODEL_DIR} not found. Run pack_model.py first." >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR/images" ]; then
    echo "> ERROR: ${INPUT_DIR}/images not found." >&2
    echo "> Populate it from benchmark_holdout.py --make-gc-test output." >&2
    exit 1
fi

"${SCRIPT_DIR}/do_build_podman.sh"

# GPU access under Podman goes through CDI. Detection probes Podman directly rather than
# querying nvidia-ctk: the specification can be present and functional while the toolkit
# binary is absent, for instance when the spec was written by hand. Probing the actual
# resolution is the only reliable test.
if podman run --rm --device nvidia.com/gpu=all \
        docker.io/library/alpine:latest true >/dev/null 2>&1; then
    GPU_ARGS=(--device nvidia.com/gpu=all)
    echo "> CDI device nvidia.com/gpu=all resolves, GPU enabled"
else
    GPU_ARGS=()
    echo "> CDI device unavailable, running on CPU (slow, but validates the plumbing)"
    echo "> Note: Grand Challenge always provides a GPU. This fallback is local only."
fi

# Restrict the container to a single GPU by default. The platform allocates one device with
# 16 GB, so exercising several cards locally would hide a memory ceiling that production will
# enforce. Override with GPU_INDEX=all to use every visible device.
GPU_INDEX="${GPU_INDEX:-0}"
if [ ${#GPU_ARGS[@]} -gt 0 ] && [ "$GPU_INDEX" != "all" ]; then
    GPU_ARGS+=(--env "CUDA_VISIBLE_DEVICES=${GPU_INDEX}")
    echo "> Restricting the container to GPU ${GPU_INDEX}"
fi

echo "> Preparing the output directory"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "> Starting the container"
# Mount flag notes:
#   :ro  read-only, matching the platform for /input and /opt/ml/model
#   :U   remap ownership to the container user; required in rootless mode so the non-root
#        user declared in the Dockerfile can write to /output
#   :z   shared SELinux relabelling, required on Fedora, RHEL and derivatives
#   --tmpfs /tmp  ephemeral scratch, mirroring the platform where /tmp is not persistent
podman run --detach \
    --name "$CONTAINER_NAME" \
    --replace \
    --platform=linux/amd64 \
    "${GPU_ARGS[@]}" \
    --volume "${MODEL_DIR}":/opt/ml/model:ro,z \
    --volume "${INPUT_DIR}":/input:ro,z \
    --volume "${OUTPUT_DIR}":/output:U,z \
    --tmpfs /tmp:size=8g \
    --shm-size=2g \
    "$IMAGE_TAG" >/dev/null

# The server is queried from INSIDE the container rather than through a published port.
# Rootless port publishing goes through pasta or slirp4netns and is frequently filtered on
# shared servers, which surfaces as an endless run of HTTP 000. Executing the request in the
# container's own network namespace removes that dependency entirely, and matches how Grand
# Challenge reaches the endpoint.
container_http() {
    # Issue an HTTP request inside the container and echo the status code.
    #
    # Inputs:  $1 method (GET or POST), $2 path, $3 timeout in seconds
    # Outputs: three-digit status code, or 000 when unreachable
    local method="$1" path="$2" timeout="$3"
    podman exec "$CONTAINER_NAME" python -c "
import sys, urllib.request, urllib.error
request = urllib.request.Request('http://127.0.0.1:${PORT}${path}', method='${method}')
if '${method}' == 'POST':
    request.data = b''
try:
    print(urllib.request.urlopen(request, timeout=${timeout}).status)
except urllib.error.HTTPError as error:
    print(error.code)
except Exception:
    print('000')
" 2>/dev/null || echo "000"
}

echo "> Waiting for the health endpoint"
for ((attempt = 1; attempt <= HEALTH_MAX_ATTEMPTS; attempt++)); do
    status=$(container_http GET /health 10)
    status="${status:-000}"
    echo "  attempt ${attempt}/${HEALTH_MAX_ATTEMPTS}: HTTP ${status}"

    if [ "$status" = "200" ]; then
        break
    fi

    # A container that has exited will never become healthy. Failing immediately surfaces the
    # startup traceback instead of waiting out the full polling budget.
    if ! podman container exists "$CONTAINER_NAME" || \
       [ "$(podman inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != "true" ]; then
        echo "> ERROR: the container stopped during startup" >&2
        podman logs "$CONTAINER_NAME" 2>&1 | tail -40
        exit 1
    fi

    if [ "$attempt" -eq "$HEALTH_MAX_ATTEMPTS" ]; then
        echo "> ERROR: the server never became healthy" >&2
        podman logs "$CONTAINER_NAME"
        exit 1
    fi
    sleep "$HEALTH_DELAY_SECONDS"
done

echo "> Calling the invoke endpoint"
start_time=$(date +%s)
status=$(container_http POST /invoke "$INVOKE_TIMEOUT_SECONDS")
elapsed=$(( $(date +%s) - start_time ))

podman logs "$CONTAINER_NAME"

if [ "$status" != "201" ]; then
    echo "> ERROR: invoke returned HTTP ${status}, expected 201" >&2
    exit 1
fi
echo "> Invoke completed in ${elapsed}s"

echo "> Verifying the outputs"
for socket in stroke-lesion-segmentation lesion-probability-map; do
    if [ ! -f "${OUTPUT_DIR}/images/${socket}/output.mha" ]; then
        echo "> ERROR: missing output for socket ${socket}" >&2
        exit 1
    fi
    echo "  ${socket}: present"
done

echo "> Checking geometry against the input"
python3 - "$INPUT_DIR" "$OUTPUT_DIR" <<'PYTHON'
"""
Compare the geometry of each prediction with the input volume.

A mismatch in size, spacing, origin or direction means the reprojection to native space is
wrong. The evaluation would then score a misaligned mask, which collapses every voxel-wise
metric without raising any error on the platform.
"""
import sys
from pathlib import Path

import SimpleITK as sitk

input_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])

reference_files = [p for d in (input_dir / "images").glob("*") if d.is_dir()
                   for p in sorted(d.glob("*.mha")) + sorted(d.glob("*.nii.gz"))]
reference = sitk.ReadImage(str(reference_files[0]))
print(f"  input : size={reference.GetSize()} spacing={tuple(round(s, 4) for s in reference.GetSpacing())}")

failed = False
for socket in ("stroke-lesion-segmentation", "lesion-probability-map"):
    prediction = sitk.ReadImage(str(output_dir / "images" / socket / "output.mha"))
    same = (
        prediction.GetSize() == reference.GetSize()
        and max(abs(a - b) for a, b in zip(prediction.GetSpacing(), reference.GetSpacing())) < 1e-4
        and max(abs(a - b) for a, b in zip(prediction.GetOrigin(), reference.GetOrigin())) < 1e-3
    )
    status = "match" if same else "MISMATCH"
    print(f"  {socket}: size={prediction.GetSize()} dtype={prediction.GetPixelIDTypeAsString()} [{status}]")
    failed = failed or not same

if failed:
    sys.exit("geometry mismatch: the prediction will not align with the reference annotation")
PYTHON

echo "> Rehearsal passed. Results in ${OUTPUT_DIR}"