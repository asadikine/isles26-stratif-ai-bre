#!/usr/bin/env bash
#
# Export the ISLES'26 submission artefacts with Podman.
#
# PIPELINE / DATA FLOW
#   [1] Rebuild the image so that the exported archive matches the current sources.
#   [2] Export the image as a docker-archive. Podman defaults to the OCI archive format,
#       which the platform rejects; --format docker-archive is mandatory.
#   [3] Pack ./model into model.tar.gz, archived from INSIDE the directory so its contents
#       land at the root of /opt/ml/model after extraction.
#   [4] Verify both archives before upload.
#
# The two archives are uploaded to two different places on Grand Challenge:
#   image     -> Your algorithm > Containers
#   model     -> Your algorithm > Models
#
# Usage: ./do_save_podman.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
IMAGE_TAG="${IMAGE_TAG:-isles26-path-a-ensemble}"
MODEL_DIR="${SCRIPT_DIR}/model"

"${SCRIPT_DIR}/do_build_podman.sh"

timestamp=$(podman inspect --format '{{ .Created }}' "$IMAGE_TAG" \
    | sed -E 's/(.*)T(.*)\..*Z?/\1_\2/' | sed 's/[:.]/-/g')
image_archive="${SCRIPT_DIR}/${IMAGE_TAG}_${timestamp}.tar.gz"
model_archive="${SCRIPT_DIR}/model.tar.gz"

echo "> Exporting the image (this takes a while)"
# --format docker-archive is not optional. Without it Podman writes an oci-archive, which the
# platform refuses at upload with an opaque error.
podman save --format docker-archive "$IMAGE_TAG" | gzip -c > "$image_archive"

echo "> Packing the model"
# -C model . archives the CONTENTS of the directory. Archiving the directory itself would
# create /opt/ml/model/model/... after extraction and the predictor would not find plans.json.
tar -czf "$model_archive" -C "$MODEL_DIR" .

echo "> Verifying the archives"
if ! tar -tzf "$image_archive" | grep -q "^manifest.json$"; then
    echo "> ERROR: the image archive has no manifest.json; it is probably an OCI archive." >&2
    echo "> Re-export with: podman save --format docker-archive" >&2
    exit 1
fi
echo "  image archive: docker-archive confirmed"

if ! tar -tzf "$model_archive" | grep -q "nnunet/"; then
    echo "> ERROR: the model archive does not start with ./nnunet/" >&2
    echo "> Re-pack with: tar -czf model.tar.gz -C model ." >&2
    exit 1
fi
echo "  model archive: layout confirmed"

echo
echo "> Upload targets on Grand Challenge:"
echo "    $(basename "$image_archive")   ->  Your algorithm > Containers"
echo "    $(basename "$model_archive")   ->  Your algorithm > Models"
echo
echo "  sizes:"
du -h "$image_archive" "$model_archive" | sed 's/^/    /'