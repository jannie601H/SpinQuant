# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

# Usage:
# bash scripts/10_optimize_rotation.sh <input_model> <w_bits> <a_bits> <kv_bits> <layerwise_flag>
# Example:
# bash scripts/10_optimize_rotation.sh meta-llama/Llama-3.2-1B 4 4 4 true
LAYERWISE_FLAG="--no-layerwise"
if [ "$5" = true ] || [ "$5" = True ] || [ "$5" = 1 ]; then
  LAYERWISE_FLAG="--layerwise"
fi

torchrun --nnodes=1 --nproc_per_node=1 optimize_rotation.py \
--input_model $1  \
--output_rotation_path "results/output_rotation/" \
--output_dir "results/output/" \
--logging_dir "results/logs" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 10 \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
$LAYERWISE_FLAG \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 64 \
--v_groupsize 64 \
