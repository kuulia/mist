#!/usr/bin/env bash
# Run neims_output_to_mgf.py for every config in this directory.
# Must be executed from the project root, e.g.:
#   bash data_processing/neims/run_all.sh

set -euo pipefail

SCRIPT="data_processing/neims/neims_output_to_mgf.py"
CONFIG_DIR="data_processing/neims"

configs=(
    #config_gecko_original.yaml
    #config_gecko_tms.yaml
    config_fc_original.yaml
    config_fc_tms.yaml
    config_kc_original.yaml
    config_kc_tms.yaml
    config_li_original.yaml
    config_li_tms.yaml
    config_wang_original.yaml
    config_wang_tms.yaml
    #config_msg_train_original.yaml
    #config_msg_val_original.yaml
    #config_msg_test_original.yaml
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

# Consolidate the three MSG split MGFs into one combined file for preprocessing.
echo "──────────────────────────────────────────"
echo "Consolidating MSG MGF splits…"
echo "──────────────────────────────────────────"
MSG_DIR="data/neims/msg/mist_inputs"
cat \
    "$MSG_DIR/msg_train_spectra.mgf" \
    "$MSG_DIR/msg_val_spectra.mgf"   \
    "$MSG_DIR/msg_test_spectra.mgf"  \
    > "$MSG_DIR/spectra_combined.mgf"
echo "Wrote $MSG_DIR/spectra_combined.mgf"

echo "──────────────────────────────────────────"
echo "Consolidating ATMOMACCS MGF splits…"
echo "──────────────────────────────────────────"
ATMOMACCS_DIR="data/neims/atmomaccs_new"
cat \
    "$ATMOMACCS_DIR/ferraz-caetano/ferraz-caetano.mgf" \
    "$ATMOMACCS_DIR/li/li.mgf"   \
    "$ATMOMACCS_DIR/wang/wang.mgf"  \
    "$ATMOMACCS_DIR/kruger-confined/kruger-confined.mgf"  \
    > "$ATMOMACCS_DIR/spectra_combined.mgf"
echo "Wrote $ATMOMACCS_DIR/spectra_combined.mgf"

echo "──────────────────────────────────────────"
echo "Consolidating ATMOMACCS TMS MGF splits…"
echo "──────────────────────────────────────────"
cat \
    "$ATMOMACCS_DIR/ferraz-caetano/ferraz-caetano_tms.mgf" \
    "$ATMOMACCS_DIR/li/li_tms.mgf"   \
    "$ATMOMACCS_DIR/wang/wang_tms.mgf"  \
    "$ATMOMACCS_DIR/kruger-confined/kruger-confined_tms.mgf"  \
    > "$ATMOMACCS_DIR/spectra_combined_tms.mgf"
echo "Wrote $ATMOMACCS_DIR/spectra_combined_tms.mgf"