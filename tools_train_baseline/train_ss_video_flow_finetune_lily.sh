#!/usr/bin/env bash
set -euo pipefail

{
export CUDA_VISIBLE_DEVICES="0,1,2,3"

EXPNAME="ss_flow_video_4d_lora_finetune_lily"
DATA_DIR="/scr/yuegao/TRELLIS_datasets/BlenderFlowers_Lilymore20kfull_sub_merged"
OUTDIR="/viscam/projects/4d-state-machine/SS4D_outputs/${EXPNAME}"

export WANDB_PROJECT="state"
export WANDB_ENTITY="ueoo-cs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
export WANDB_NAME="$(basename "$OUTDIR")__${RUN_TS}"

MASTER_PORT=$((20000 + RANDOM % 20000))

python train.py \
    --config configs/baseline_video_finetune/ss_flow_video_4d_lora_finetune_lily.json \
    --output_dir "$OUTDIR" \
    --data_dir "$DATA_DIR" \
    --master_port "$MASTER_PORT"

exit 0
}
