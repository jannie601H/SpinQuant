#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [ "$#" -lt 7 ] || [ "$#" -gt 9 ]; then
    echo "Usage: $0 MODEL W_BITS A_BITS K_BITS V_BITS {gptq|rtn} ROTATION [R3] [R4]" >&2
    echo "Rotation switches: on/off. Defaults: R3 on for K<16, R4 follows ROTATION." >&2
    echo "Environment: MAX_STEPS=10, ROTATION_SEED=0, FORCE_ROTATION=0, RESULT_DIR=results" >&2
    exit 1
fi

MODEL=$1
W_BITS=$2
A_BITS=$3
K_BITS=$4
V_BITS=$5
QUANTIZER=$6
ROTATION=$7
MAX_STEPS=${MAX_STEPS:-10}
ROTATION_SEED=${ROTATION_SEED:-0}
FORCE_ROTATION=${FORCE_ROTATION:-0}
RESULT_DIR=${RESULT_DIR:-results}

for bits in "$W_BITS" "$A_BITS" "$K_BITS" "$V_BITS"; do
    if ! [[ "$bits" =~ ^([1-9]|1[0-6])$ ]]; then
        echo "ERROR: W/A/K/V bits must be integers from 1 to 16" >&2
        exit 1
    fi
done
if ! [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$ROTATION_SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: MAX_STEPS must be positive; ROTATION_SEED must be nonnegative" >&2
    exit 1
fi
if [[ "$QUANTIZER" != gptq && "$QUANTIZER" != rtn ]]; then
    echo "ERROR: QUANTIZER must be gptq or rtn" >&2
    exit 1
fi
if [[ "$FORCE_ROTATION" != 0 && "$FORCE_ROTATION" != 1 ]]; then
    echo "ERROR: FORCE_ROTATION must be 0 or 1" >&2
    exit 1
fi

# Preserve previous behavior when the optional switches are omitted.
R3_DEFAULT=off
if [ "$K_BITS" -lt 16 ]; then R3_DEFAULT=on; fi
R3=${8:-$R3_DEFAULT}
R4=${9:-$ROTATION}
for setting in "$ROTATION" "$R3" "$R4"; do
    if [[ "$setting" != on && "$setting" != off ]]; then
        echo "ERROR: ROTATION, R3 and R4 must be on or off" >&2
        exit 1
    fi
done
ROT_FLAGS=()
if [ "$R3" = on ]; then ROT_FLAGS+=(--r3); else ROT_FLAGS+=(--no-r3); fi
if [ "$R4" = on ]; then ROT_FLAGS+=(--r4); else ROT_FLAGS+=(--no-r4); fi

# Shared options keep optimization and evaluation quantization consistent.
COMMON_ARGS=(
    --input_model "$MODEL"
    --model_max_length 2048
    --fp16 False
    --bf16 True
    --save_safetensors False
    --seed "$ROTATION_SEED"
    --w_bits "$W_BITS"
    --a_bits "$A_BITS"
    --k_bits "$K_BITS"
    --v_bits "$V_BITS"
    --w_clip
    --a_asym
    --k_asym
    --v_asym
    --k_groupsize 64
    --v_groupsize 64
    "${ROT_FLAGS[@]}"
)

mkdir -p "$RESULT_DIR"
MODEL_NAME=$(basename "$MODEL")
EXPERIMENT="${MODEL_NAME}_W${W_BITS}A${A_BITS}K${K_BITS}V${V_BITS}_rot-${ROTATION}_r3-${R3}_r4-${R4}"

echo "Model      : $MODEL"
echo "W/A/K/V    : $W_BITS / $A_BITS / $K_BITS / $V_BITS"
echo "Quantizer  : $QUANTIZER"
echo "R1/R2      : $ROTATION"
echo "R3 / R4    : $R3 / $R4"
echo "Max steps  : $MAX_STEPS"
echo "Seed       : $ROTATION_SEED"

CMD=(torchrun --nnodes=1 --nproc_per_node=1 ptq.py
    "${COMMON_ARGS[@]}"
    --do_train False --do_eval True --per_device_eval_batch_size 1)
if [ "$QUANTIZER" = rtn ]; then CMD+=(--w_rtn); fi

if [ "$ROTATION" = on ]; then
    # max_steps affects both training length and the cosine LR schedule.
    # The helper fingerprints all training arguments, source and package versions.
    OPT_CMD=(torchrun --nnodes=1 --nproc_per_node=1 optimize_rotation.py
        "${COMMON_ARGS[@]}"
        --per_device_train_batch_size 1
        --max_steps "$MAX_STEPS"
        --learning_rate 1.5
        --weight_decay 0.0
        --lr_scheduler_type cosine
        --gradient_checkpointing True
        --logging_steps 1
        --log_on_each_node False
        --save_strategy no)
    CACHE_ARGS=(--cache-root "$RESULT_DIR/rotation")
    if [ "$FORCE_ROTATION" = 1 ]; then CACHE_ARGS+=(--force); fi
    ROT_PATH=$(python "$SCRIPT_DIR/rotation_cache.py" "${CACHE_ARGS[@]}" -- "${OPT_CMD[@]}")
    CMD+=(--rotate --optimized_rotation_path "$ROT_PATH")
    # Include the cache key so different training settings cannot share a PTQ log.
    EXPERIMENT=$(basename "$(dirname "$ROT_PATH")")
fi

LOG_PATH="$RESULT_DIR/${EXPERIMENT}_${QUANTIZER}_seed-${ROTATION_SEED}.log"
echo "Log file   : $LOG_PATH"
echo "Running PTQ evaluation..."
"${CMD[@]}" 2>&1 | tee "$LOG_PATH"
