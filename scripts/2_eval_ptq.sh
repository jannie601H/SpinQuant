# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

# Usage:
# bash scripts/2_eval_ptq.sh <input_model> <w_bits> <a_bits> <kv_bits> <layerwise_flag>
# Example:
# bash scripts/2_eval_ptq.sh meta-llama/Llama-3.2-1B 4 4 4 true
LAYERWISE_FLAG="--no-layerwise"
if [ "$5" = true ] || [ "$5" = True ] || [ "$5" = 1 ]; then
  LAYERWISE_FLAG="--layerwise"
fi

torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 64 \
--v_groupsize 64 \
--rotate \
$LAYERWISE_FLAG \
--optimized_rotation_path "results/output_rotation/R.bin" \

