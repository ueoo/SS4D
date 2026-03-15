#!/usr/bin/env bash
# Render SS4D from_wan inference results with GT data camera view (same pattern as AnimateAnyMesh infer_trellis_sample_dahliafull).
set -euo pipefail

{

DATA_ROOT="${DATA_ROOT:-/scr/yuegao/ss4d_datasets_from_wan}"
OUTPUT_BASE="${OUTPUT_BASE:-/scr/yuegao/ss4d_inference_outputs_from_wan_finetune_step2}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/scr/yuegao/TRELLIS_datasets/BlenderFlowers_Dahliamorefix20kfullTest_merged}"
PIPELINE_PATH="${PIPELINE_PATH:-lizb6626/SS4D}"
INPUT_VIEW_IDX=0
SEED="${SEED:-42}"
MAX_FRAMES=24
FRAME_STEP=2
T_END=048

SCENE_NAME="Dahliamorefix_020033"
CAMERA_VIEW_IDX="${CAMERA_VIEW_IDX:-7}"

PY_SCRIPT="test_ours_dataset.py"

run_sample() {
  local sample_name="$1"
  local frame="$2"
  local transforms_path="${TEST_DATA_ROOT}/renders/${SCENE_NAME}_${frame}/transforms.json"
  [[ -f "${transforms_path}" ]] || { echo "Missing transforms: ${transforms_path}"; return 1; }
  echo "Rendering SS4D from_wan: ${sample_name} (view ${CAMERA_VIEW_IDX} from ${transforms_path})"
  python3 "${PY_SCRIPT}" \
    --data_root "${DATA_ROOT}" \
    --sample_name "${sample_name}" \
    --input_view_idx "${INPUT_VIEW_IDX}" \
    --render_transforms_path "${transforms_path}" \
    --render_view_idx "${CAMERA_VIEW_IDX}" \
    --max_frames "${MAX_FRAMES}" \
    --frame_step "${FRAME_STEP}" \
    --pipeline_path "${PIPELINE_PATH}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_BASE}/${sample_name}"
}

run_sample "Dahliamorefix_020033_010_from_wan_forward_s000_t${T_END}" "010"
run_sample "Dahliamorefix_020033_100_from_wan_reverse_s000_t${T_END}" "100"
echo "Done. Outputs under ${OUTPUT_BASE}"

exit 0
}
