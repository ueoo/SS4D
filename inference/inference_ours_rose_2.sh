#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/scr/yuegao/ss4d_datasets"
OUTPUT_BASE="/scr/yuegao/ss4d_inference_outputs"
PIPELINE_PATH="lizb6626/SS4D"
INPUT_VIEW_IDX=8
RENDER_VIEW_IDXS="6"
MAX_FRAMES=""
RESOLUTION=""
SEED=42
SAVE_METADATA=0
RUN_BOTH_DIRECTIONS=1

SAMPLE_BASE="Rosemore_050047"
FRAME_START=69
FRAME_END=179

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/test_ours_dataset.py"

run_one_direction() {
  local direction="$1"
  local sample_name="${SAMPLE_BASE}_${direction}_s$(printf "%03d" "${FRAME_START}")_t$(printf "%03d" "${FRAME_END}")"
  local output_dir="${OUTPUT_BASE}/${sample_name}"
  local cmd=(
    python3 "${PY_SCRIPT}"
    --data_root "${DATA_ROOT}"
    --sample_name "${sample_name}"
    --input_view_idx "${INPUT_VIEW_IDX}"
    --render_view_idxs "${RENDER_VIEW_IDXS}"
    --pipeline_path "${PIPELINE_PATH}"
    --seed "${SEED}"
    --output_dir "${output_dir}"
  )
  if [[ -n "${MAX_FRAMES}" ]]; then
    cmd+=(--max_frames "${MAX_FRAMES}")
  fi
  if [[ -n "${RESOLUTION}" ]]; then
    cmd+=(--resolution "${RESOLUTION}")
  fi
  if [[ "${SAVE_METADATA}" == "1" ]]; then
    cmd+=(--save_metadata)
  fi
  echo "Running ${direction} inference:"
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

if [[ "${RUN_BOTH_DIRECTIONS}" == "1" ]]; then
  run_one_direction "forward"
  run_one_direction "reverse"
else
  run_one_direction "reverse"
fi
