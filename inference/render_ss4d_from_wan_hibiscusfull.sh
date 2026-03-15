#!/usr/bin/env bash
# Render SS4D from_wan inference results with GT data camera view (same pattern as AnimateAnyMesh infer_trellis_sample_hibiscusfull).
set -euo pipefail

{

DATA_ROOT="${DATA_ROOT:-/scr/yuegao/ss4d_datasets_from_wan}"
OUTPUT_BASE="${OUTPUT_BASE:-/scr/yuegao/ss4d_inference_outputs_from_wan_finetune_step2}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/scr/yuegao/TRELLIS_datasets/BlenderFlowers_Hibiscusmore10kfullTest_merged}"
PIPELINE_PATH="${PIPELINE_PATH:-lizb6626/SS4D}"
INPUT_VIEW_IDX=0
SEED="${SEED:-42}"
MAX_FRAMES=24
FRAME_STEP=2
T_END=048

SMALL_SCENE_NAME="Hibiscusmore_010054"
LARGE_SCENE_NAME="Hibiscusmore_010008"
SMALL_FRAME="020"
LARGE_FRAME="070"
SMALL_CAMERA_VIEW_IDX="${SMALL_CAMERA_VIEW_IDX:-3}"
LARGE_CAMERA_VIEW_IDX="${LARGE_CAMERA_VIEW_IDX:-4}"

PY_SCRIPT="test_ours_dataset.py"

run_sample() {
  local sample_name="$1"
  local scene_name="$2"
  local frame="$3"
  local view_idx="$4"
  local transforms_path="${TEST_DATA_ROOT}/renders/${scene_name}_${frame}/transforms.json"
  [[ -f "${transforms_path}" ]] || { echo "Missing transforms: ${transforms_path}"; return 1; }
  echo "Rendering SS4D from_wan: ${sample_name} (view ${view_idx} from ${transforms_path})"
  python3 "${PY_SCRIPT}" \
    --data_root "${DATA_ROOT}" \
    --sample_name "${sample_name}" \
    --input_view_idx "${INPUT_VIEW_IDX}" \
    --render_transforms_path "${transforms_path}" \
    --render_view_idx "${view_idx}" \
    --max_frames "${MAX_FRAMES}" \
    --frame_step "${FRAME_STEP}" \
    --pipeline_path "${PIPELINE_PATH}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_BASE}/${sample_name}"
}

run_sample "Hibiscusmore_010054_020_from_wan_forward_s000_t${T_END}" "${SMALL_SCENE_NAME}" "${SMALL_FRAME}" "${SMALL_CAMERA_VIEW_IDX}"
run_sample "Hibiscusmore_010008_070_from_wan_reverse_s000_t${T_END}" "${LARGE_SCENE_NAME}" "${LARGE_FRAME}" "${LARGE_CAMERA_VIEW_IDX}"
echo "Done. Outputs under ${OUTPUT_BASE}"

exit 0
}
