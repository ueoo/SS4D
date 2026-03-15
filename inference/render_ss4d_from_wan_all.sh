#!/usr/bin/env bash
# Render all SS4D from_wan results with GT data view index; run scripts in parallel on different CUDA devices.
set -euo pipefail

{
SCRIPT_DIR="inference"

# Run each flower script in the background on its own GPU (0–4). Adjust if you have fewer GPUs.
echo "Launching render_ss4d_from_wan_* on CUDA 0–4..."
CUDA_VISIBLE_DEVICES=0 bash "${SCRIPT_DIR}/render_ss4d_from_wan_dahliafull.sh" &
CUDA_VISIBLE_DEVICES=1 bash "${SCRIPT_DIR}/render_ss4d_from_wan_daisyfull.sh" &
CUDA_VISIBLE_DEVICES=2 bash "${SCRIPT_DIR}/render_ss4d_from_wan_hibiscusfull.sh" &
CUDA_VISIBLE_DEVICES=3 bash "${SCRIPT_DIR}/render_ss4d_from_wan_lilyfull.sh" &
# CUDA_VISIBLE_DEVICES=4 bash "${SCRIPT_DIR}/render_ss4d_from_wan_rosefull.sh" &

wait
echo "All render_ss4d_from_wan done."

exit 0
}
