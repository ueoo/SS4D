#!/usr/bin/env bash
# SS4D inference on WAN I2V lily. (1) Lilymore_020098 fwd; (2) Lilymore_020016 rev. Data from create_ss4d_from_wan_lily.sh.
set -euo pipefail

{
DATA_ROOT="/scr/yuegao/ss4d_datasets_from_wan"
OUTPUT_BASE="/scr/yuegao/ss4d_inference_outputs_from_wan"
PIPELINE_PATH="lizb6626/SS4D"
INPUT_VIEW_IDX=0
RENDER_VIEW_IDXS="0"
SEED=42
MAX_FRAMES=32
FRAME_STEP=1
T_END=48

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/test_ours_dataset.py"

run_sample() {
  local sample_name="$1"
  echo "Running SS4D from_wan: ${sample_name}"
  python3 "${PY_SCRIPT}" --data_root "${DATA_ROOT}" --sample_name "${sample_name}" --input_view_idx "${INPUT_VIEW_IDX}" --render_view_idxs "${RENDER_VIEW_IDXS}" --max_frames "${MAX_FRAMES}" --frame_step "${FRAME_STEP}" --pipeline_path "${PIPELINE_PATH}" --seed "${SEED}" --output_dir "${OUTPUT_BASE}/${sample_name}"
}

run_sample "Lilymore_020098_011_from_wan_forward_s000_t${T_END}"
run_sample "Lilymore_020016_099_from_wan_reverse_s000_t${T_END}"
echo "Done. Outputs under ${OUTPUT_BASE}"

exit 0
}
