#!/usr/bin/env bash
# Download the TMS-derivatized ATMOMACCS data from the Hugging Face dataset
# atmospheic-ei-ms-team/atmospheric-ei-ms into the layout config_*_tms.yaml expects.
# (Yes, the org slug is missing an 'r' — that is the actual namespace on HF.)
#
# Must be executed from the project root, e.g.:
#   HF_TOKEN=hf_xxx bash data_processing/neims/00_download_atmomaccs_tms.sh
#
# The repo is private, so a read token is required.  It is taken from, in order:
#   $HF_TOKEN, $HUGGING_FACE_HUB_TOKEN, ~/.cache/huggingface/token
#
# Resulting layout (mirrors the "original" splits already in atmomaccs_new/):
#   data/neims/atmomaccs_new/<folder>/compounds/derivatized/...
#   data/neims/atmomaccs_new/<folder>/NEIMS/derivatized/spectra.sdf

set -euo pipefail

# Override with HF_REPO=... if the org slug is ever corrected upstream.
REPO="${HF_REPO:-atmospheic-ei-ms-team/atmospheric-ei-ms}"
REVISION="${HF_REVISION:-main}"
DEST_ROOT="data/neims/atmomaccs_new"

# The four datasets that make up ATMOMACCS (HF folder names).
folders=(
    ferraz-caetano
    kruger-confined
    li
    wang
)

# ── Token ────────────────────────────────────────────────────────────────────
TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [[ -z "$TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    TOKEN="$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")"
fi
if [[ -z "$TOKEN" ]]; then
    echo "ERROR: no Hugging Face token found." >&2
    echo "       $REPO is private; export HF_TOKEN=hf_... and re-run." >&2
    exit 1
fi

# Fail early with a clear message if the token cannot read the repo.
status="$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $TOKEN" \
    "https://huggingface.co/api/datasets/$REPO")"
if [[ "$status" != "200" ]]; then
    echo "ERROR: cannot read $REPO (HTTP $status)." >&2
    echo "       401 → token missing or malformed." >&2
    echo "       404 → token is valid but cannot see this repo.  A fine-grained" >&2
    echo "             token needs explicit read access to the org, and your" >&2
    echo "             account must be a member of it." >&2
    exit 1
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
# List the files under data/<folder>/ that live in a 'derivatized' directory.
list_derivatized() {
    local folder="$1"
    curl -sf -H "Authorization: Bearer $TOKEN" \
        "https://huggingface.co/api/datasets/$REPO/tree/$REVISION/data/$folder?recursive=true" \
    | python3 -c '
import json, sys
for e in json.load(sys.stdin):
    if e.get("type") == "file" and "/derivatized/" in e["path"] + "/":
        print(e["path"])
'
}

n_files=0

for folder in "${folders[@]}"; do
    echo "──────────────────────────────────────────"
    echo "Dataset: $folder"
    echo "──────────────────────────────────────────"

    mapfile -t paths < <(list_derivatized "$folder")
    if (( ${#paths[@]} == 0 )); then
        echo "ERROR: no derivatized files listed for $folder" >&2
        exit 1
    fi

    for path in "${paths[@]}"; do
        # path looks like: data/<folder>/NEIMS/derivatized/spectra.sdf
        dest="$DEST_ROOT/${path#data/}"
        mkdir -p "$(dirname "$dest")"
        echo "  → $dest"
        curl -fL --progress-bar -H "Authorization: Bearer $TOKEN" \
            -o "$dest" \
            "https://huggingface.co/datasets/$REPO/resolve/$REVISION/$path"
        (( n_files++ )) || true
    done
done

echo "══════════════════════════════════════════"
echo "Downloaded $n_files files into $DEST_ROOT"
echo
echo "Next:"
echo "  bash data_processing/neims/run_all.sh          # SDF → MGF"
echo "  bash data_processing/neims/run_preprocessing.sh"
