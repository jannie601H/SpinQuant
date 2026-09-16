#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [ "$#" -lt 7 ] || [ "$#" -gt 8 ]; then
    echo "Usage: $0 MODEL W_BITS A_BITS K_BITS V_BITS {gptq|rtn} ROTATION [HAD]" >&2
    echo "HAD defaults to ROTATION. HAD=on enables R4 and enables R3 only for K<16." >&2
    echo "Environment: PYTHON_BIN=python3, RESPIN=0, MAX_STEPS=10, ROTATION_SEED=0, FORCE_ROTATION=0, RESULT_DIR=results" >&2
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
RESPIN=${RESPIN:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}

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
if [[ "$RESPIN" != 0 && "$RESPIN" != 1 ]]; then
    echo "ERROR: RESPIN must be 0 or 1" >&2
    exit 1
fi
if [[ "$RESPIN" = 1 && "$ROTATION" != on ]]; then
    echo "ERROR: RESPIN=1 requires ROTATION=on" >&2
    exit 1
fi

# No Rotation disables all rotations; HAD controls the online Hadamard pair.
HAD=${8:-$ROTATION}
for setting in "$ROTATION" "$HAD"; do
    if [[ "$setting" != on && "$setting" != off ]]; then
        echo "ERROR: ROTATION and HAD must be on or off" >&2
        exit 1
    fi
done
if [[ "$ROTATION" = off && "$HAD" = on ]]; then
    echo "ERROR: No Rotation requires HAD=off" >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python not found: $PYTHON_BIN. Set PYTHON_BIN to your environment's Python executable." >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c 'import torch.distributed.run' >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN cannot import torch.distributed.run. Activate the training environment or set PYTHON_BIN to its Python executable." >&2
    exit 1
fi
R3=off
R4=$HAD
if [[ "$HAD" = on && "$K_BITS" -lt 16 ]]; then R3=on; fi
if [ "$QUANTIZER" = gptq ]; then
    ROT_W_BITS=16
else
    ROT_W_BITS=$W_BITS
fi
ROT_FLAGS=()
if [ "$RESPIN" = 1 ]; then ROT_FLAGS+=(--respin); fi
if [ "$R3" = on ]; then ROT_FLAGS+=(--r3); else ROT_FLAGS+=(--no-r3); fi
if [ "$R4" = on ]; then ROT_FLAGS+=(--r4); else ROT_FLAGS+=(--no-r4); fi

# A/K/V are shared; weight precision is deliberately stage-specific.
COMMON_ARGS=(
    --input_model "$MODEL"
    --model_max_length 2048
    --fp16 False
    --bf16 True
    --save_safetensors False
    --seed "$ROTATION_SEED"
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
echo "Model      : $MODEL"
echo "Python     : $PYTHON_BIN"
echo "Target W/A/K/V : $W_BITS / $A_BITS / $K_BITS / $V_BITS"
echo "Quantizer  : $QUANTIZER"
echo "R1/R2      : $ROTATION"
echo "Respin A/B : $RESPIN"
echo "Had        : $HAD"
echo "R3 / R4    : $R3 / $R4"
echo "Max steps  : $MAX_STEPS"
echo "Seed       : $ROTATION_SEED"

CMD=("$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --nproc_per_node=1 ptq.py
    "${COMMON_ARGS[@]}"
    --w_bits "$W_BITS"
    --do_train False --do_eval True --per_device_eval_batch_size 1)
if [ "$QUANTIZER" = rtn ]; then CMD+=(--w_rtn); fi

if [ "$ROTATION" = on ]; then
    echo "Rotation optimization W/A/K/V : $ROT_W_BITS / $A_BITS / $K_BITS / $V_BITS"
    # max_steps affects both training length and the cosine LR schedule.
    # The helper fingerprints all training arguments, source and package versions.
    OPT_CMD=("$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --nproc_per_node=1 optimize_rotation.py
        "${COMMON_ARGS[@]}"
        --w_bits "$ROT_W_BITS"
        --per_device_train_batch_size 1
        --max_steps "$MAX_STEPS"
        --learning_rate 15
        --weight_decay 0.0
        --lr_scheduler_type cosine
        --gradient_checkpointing True
        --logging_steps 1
        --log_on_each_node False
        --save_strategy no)
    CACHE_ARGS=(--cache-root "$RESULT_DIR/rotation")
    if [ "$FORCE_ROTATION" = 1 ]; then CACHE_ARGS+=(--force); fi
    ROT_PATH=$("$PYTHON_BIN" "$SCRIPT_DIR/rotation_cache.py" "${CACHE_ARGS[@]}" -- "${OPT_CMD[@]}")
    CMD+=(--rotate --optimized_rotation_path "$ROT_PATH")
fi

# Final results describe target precision, independently of the rotation cache.
LOG_PATH=$("$PYTHON_BIN" "$SCRIPT_DIR/rotation_cache.py" --ptq-result-dir "$RESULT_DIR" -- "${CMD[@]}")
echo "Log file   : $LOG_PATH"
echo "Running PTQ evaluation..."
{
    echo "Model=$MODEL target_W=$W_BITS A=$A_BITS K=$K_BITS V=$V_BITS quantizer=$QUANTIZER"
    echo "rotation=$ROTATION respin=$RESPIN had=$HAD r3=$R3 r4=$R4 seed=$ROTATION_SEED"
    if [ "$ROTATION" = on ]; then
        echo "rotation_w_bits=$ROT_W_BITS max_steps=$MAX_STEPS checkpoint=$ROT_PATH"
    else
        echo "Rotation optimization skipped"
    fi
    "${CMD[@]}"
} 2>&1 | tee "$LOG_PATH"
