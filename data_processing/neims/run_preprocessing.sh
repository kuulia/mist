#!/usr/bin/env bash
# Run neims_preprocessing.py for every dataset.
# Must be executed from the project root, e.g.:
#   bash data_processing/neims/run_preprocessing.sh
#
# Prerequisites: run run_all.sh first to produce the MGF files, and for MSG
# also cat the three split MGFs into spectra_combined.mgf (see config_msg_preprocess.yaml).

set -euo pipefail

SCRIPT="data_processing/neims/neims_preprocessing.py"
CONFIG_DIR="data_processing/neims"

configs=(
    config_gecko_preprocess.yaml
    config_gecko_tms_preprocess.yaml
    config_fc_preprocess.yaml
    config_fc_tms_preprocess.yaml
    config_kc_preprocess.yaml
    config_kc_tms_preprocess.yaml
    config_li_preprocess.yaml
    config_li_tms_preprocess.yaml
    config_wang_preprocess.yaml
    config_wang_tms_preprocess.yaml
    config_msg_preprocess.yaml
)

n_ok=0
n_fail=0
failed=()

for cfg in "${configs[@]}"; do
    echo "──────────────────────────────────────────"
    echo "Running: $cfg"
    echo "──────────────────────────────────────────"
    if python "$SCRIPT" "$CONFIG_DIR/$cfg"; then
        (( n_ok++ )) || true
    else
        echo "FAILED: $cfg" >&2
        (( n_fail++ )) || true
        failed+=("$cfg")
    fi
done

echo "══════════════════════════════════════════"
echo "Done: $n_ok succeeded, $n_fail failed"
if (( n_fail > 0 )); then
    echo "Failed configs:"
    for cfg in "${failed[@]}"; do echo "  - $cfg"; done
    exit 1
fi
