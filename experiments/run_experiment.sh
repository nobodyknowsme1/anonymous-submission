#!/bin/bash
# Common experiment runner for CoSA
# This script runs training and evaluation for different methods

# Parse command line arguments
METHOD=${1}
SEED=${2:-42}
MODEL_NAME=${3:-llama31-8b}

# Use seed as run ID for uniqueness
RUN_ID=${SEED}

# Show usage if no arguments provided
if [ -z "$METHOD" ]; then
    echo "Usage: $0 [METHOD] [SEED] [MODEL]"
    echo ""
    echo "Arguments:"
    echo "  METHOD    Training method: lora, pissa, compressed, adalora, full"
    echo "  SEED      Random seed (also serves as run ID) (default: 42)"
    echo "  MODEL     Model short name or full path (default: llama31-8b)"
    echo ""
    echo "Predefined model shortcuts:"
    echo "  llama31-8b    -> meta-llama/Llama-3.1-8B"
    echo "  llama32-1b    -> meta-llama/Llama-3.2-1B"
    echo "  llama2-7b     -> meta-llama/Llama-2-7b-hf"
    echo "  qwen2-7b      -> Qwen/Qwen2-7B"
    echo ""
    echo "Example:"
    echo "  $0 lora 42 llama32-1b"
    echo "  $0 pissa 123 meta-llama/custom-model"
    echo ""
    echo "Environment variables (optional overrides):"
    echo "  BASE_MODEL    Model to fine-tune (overrides MODEL argument)"
    echo "  DATA_PATH     Dataset location (default: pissa-dataset)"
    echo "  OUTPUT_PATH   Output directory (default: auto-generated)"
    echo "  GPUS          GPU IDs to use (default: 0,1,2,3)"
    echo "  BATCH_SIZE    Batch size per device"
    echo "  LEARNING_RATE Learning rate"
    echo "  LORA_RANK     LoRA rank (for lora/pissa/compressed)"
    echo "  LORA_ALPHA    LoRA alpha (for lora/pissa/compressed)"
    exit 1
fi

# Map short model names to full paths
resolve_model_name() {
    case "$1" in
        llama31-8b)
            echo "meta-llama/Llama-3.1-8B"
            ;;
        llama32-1b)
            echo "meta-llama/Llama-3.2-1B"
            ;;
        llama2-7b)
            echo "meta-llama/Llama-2-7b-hf"
            ;;
        qwen2-7b)
            echo "Qwen/Qwen2-7B"
            ;;
        *)
            # If not a predefined shortcut, assume it's a full path
            echo "$1"
            ;;
    esac
}

# Set default configurations if not already set
# If BASE_MODEL is not set, resolve the MODEL_NAME argument
if [ -z "$BASE_MODEL" ]; then
    BASE_MODEL=$(resolve_model_name "$MODEL_NAME")
fi
DATA_PATH=${DATA_PATH:-"pissa-dataset"}

# Generate default output path based on method and run ID
if [ -z "$OUTPUT_PATH" ]; then
    case $METHOD in
        lora)
            OUTPUT_PATH="output/metamath-lora-${BASE_MODEL##*/}-r${LORA_RANK:-128}-a${LORA_ALPHA:-128}-run${RUN_ID}"
            ;;
        pissa)
            OUTPUT_PATH="output/metamath-pissa-${BASE_MODEL##*/}-r${LORA_RANK:-128}-a${LORA_ALPHA:-128}-run${RUN_ID}"
            ;;
        compressed)
            OUTPUT_PATH="output/metamath-compressed-${BASE_MODEL##*/}-a${COMPRESSION_A:-1024}b${COMPRESSION_B:-256}-run${RUN_ID}"
            ;;
        adalora)
            OUTPUT_PATH="output/metamath-adalora-${BASE_MODEL##*/}-ir${INIT_R:-160}-tr${TARGET_R:-64}-a${ADALORA_A:-128}-run${RUN_ID}"
            ;;
        full)
            OUTPUT_PATH="output/metamath-full-${BASE_MODEL##*/}-run${RUN_ID}"
            ;;
    esac
fi

# Optional variables with defaults
GPUS=${GPUS:-"0,1,2,3"}
NUM_EPOCHS=${NUM_EPOCHS:-1}
MAX_LENGTH=${MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
SUB_TASK=${SUB_TASK:-"metamath:100000"}
DATASET_FIELD=${DATASET_FIELD:-"instruction output"}
EVAL_TASK=${EVAL_TASK:-"metamath"}

# Method-specific batch size adjustments
if [ "$METHOD" == "full" ]; then
    BATCH_SIZE=${FULL_BATCH_SIZE:-1}
elif [ "$METHOD" == "lora" ] || [ "$METHOD" == "pissa" ]; then
    BATCH_SIZE=${LORA_BATCH_SIZE:-4}
fi

# Common training arguments
COMMON_ARGS="
    --deepspeed configs/ds_config_zero2_no_offload.json
    --model_name_or_path $BASE_MODEL
    --bf16
    --data_path $DATA_PATH
    --sub_task $SUB_TASK
    --dataset_split train
    --dataset_field $DATASET_FIELD
    --output_dir $OUTPUT_PATH
    --num_train_epochs $NUM_EPOCHS
    --model_max_length $MAX_LENGTH
    --per_device_train_batch_size $BATCH_SIZE
    --gradient_accumulation_steps $GRAD_ACCUM
    --save_strategy steps
    --save_steps 1000
    --save_total_limit 1
    --learning_rate $LEARNING_RATE
    --lr_scheduler_type cosine
    --report_to tensorboard
    --seed $SEED"

# Default common args that may be overridden by methods
DEFAULT_ARGS="
    --weight_decay 0.
    --warmup_ratio 0.03
    --logging_steps 1
    --merge True"

# Set up environment
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# Change to repository root if needed (assumes script is in experiments/ subdirectory)
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
if [ "$(pwd)" != "$REPO_ROOT" ]; then
    cd "$REPO_ROOT"
fi

echo "Running $METHOD experiment - Seed/Run $SEED"
echo "Using model: $BASE_MODEL"

# Method-specific configurations and training
case $METHOD in
    lora)
        SCRIPT="train.py"
        METHOD_ARGS="
            --full_finetune False
            --init_weights True
            --lora_rank ${LORA_RANK:-128}
            --lora_alpha ${LORA_ALPHA:-128}
            --target_modules ${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
            --learning_rate ${LEARNING_RATE}"

        deepspeed --master_port=$((16970 + RUN_ID)) --include=localhost:$GPUS \
            $SCRIPT $COMMON_ARGS $DEFAULT_ARGS $METHOD_ARGS
        ;;

    pissa)
        SCRIPT="train.py"

        # PiSSA initialization if needed
        RES_MODEL=${RES_MODEL:-"${OUTPUT_PATH%/*}/PiSSA-${BASE_MODEL##*/}-r${LORA_RANK:-128}-run${RUN_ID}"}
        if [ ! -e "$RES_MODEL" ]; then
            echo "Performing PiSSA initialization..."
            python utils/init_pissa.py \
                --base_model_path $BASE_MODEL \
                --output_dir $RES_MODEL \
                --init_weights pissa_niter_16 \
                --lora_r ${LORA_RANK:-128} \
                --lora_alpha ${LORA_ALPHA:-128} \
                --lora_dropout 0 \
                --target_modules ${TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}
        else
            echo "Using existing PiSSA residual model: $RES_MODEL"
        fi

        METHOD_ARGS="
            --full_finetune False
            --adapter_name_or_path pissa_init
            --learning_rate ${LEARNING_RATE}"

        # Update model path for PiSSA
        COMMON_ARGS="${COMMON_ARGS/--model_name_or_path $BASE_MODEL/--model_name_or_path $RES_MODEL}"

        deepspeed --master_port=$((16970 + RUN_ID)) --include=localhost:$GPUS \
            $SCRIPT $COMMON_ARGS $DEFAULT_ARGS $METHOD_ARGS
        ;;

    compressed)
        SCRIPT="train_compressed.py"
        METHOD_ARGS="
            --full_finetune False
            --lora_rank ${LORA_RANK:-128}
            --lora_alpha ${LORA_ALPHA:-128}
            --compression_a ${COMPRESSION_A:-1024}
            --compression_b ${COMPRESSION_B:-256}
            --max_grad_norm 1.0
            --learning_rate ${LEARNING_RATE}"

        deepspeed --master_port=$((16970 + RUN_ID)) --include=localhost:$GPUS \
            $SCRIPT $COMMON_ARGS $DEFAULT_ARGS $METHOD_ARGS
        ;;

    adalora)
        SCRIPT="train.py"
        METHOD_ARGS="
            --full_finetune False
            --use_adalora
            --adalora_init_r ${INIT_R:-160}
            --adalora_a ${ADALORA_A:-128}
            --adalora_target_r ${TARGET_R:-64}
            --target_modules ${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
            --learning_rate ${ADALORA_LR:-2e-4}"

        deepspeed --master_port=$((16970 + RUN_ID)) --include=localhost:$GPUS \
            $SCRIPT $COMMON_ARGS $DEFAULT_ARGS $METHOD_ARGS
        ;;

    full)
        SCRIPT="train.py"
        METHOD_ARGS="
            --full_finetune True
            --learning_rate ${FULL_LR:-1e-5}
            --weight_decay ${FULL_WD:-0.01}
            --warmup_steps ${FULL_WARMUP:-200}
            --adam_beta1 ${ADAM_BETA1:-0.9}
            --adam_beta2 ${ADAM_BETA2:-0.995}
            --adam_epsilon ${ADAM_EPSILON:-1e-8}
            --max_grad_norm ${FULL_MAX_GRAD:-0.5}
            --gradient_checkpointing ${GRADIENT_CHECKPOINTING:-True}
            --logging_steps ${FULL_LOG_STEPS:-10}
            --num_train_epochs ${FULL_EPOCHS:-3}
            --save_steps ${FULL_SAVE_STEPS:-1000}"

        # Override common args for full fine-tuning
        BATCH_SIZE=${FULL_BATCH_SIZE:-1}
        GRAD_ACCUM=${FULL_GRAD_ACCUM:-8}

        # For full fine-tuning, METHOD_ARGS overrides some defaults
        deepspeed --master_port=$((16970 + RUN_ID)) --include=localhost:$GPUS \
            $SCRIPT $COMMON_ARGS $METHOD_ARGS
        ;;

    *)
        echo "Error: Unknown method '$METHOD'"
        echo "Valid methods: lora, pissa, compressed, adalora, full"
        exit 1
        ;;
esac

# Run evaluation if training succeeded
if [ $? -eq 0 ]; then
    echo "Running evaluation for $OUTPUT_PATH..."

    # Ensure VLLM uses the correct NCCL library path
    export VLLM_NCCL_SO_PATH="$CONDA_PREFIX/lib/libnccl.so.2"

    # Simple evaluation using the utility scripts
    python utils/gen_vllm.py \
        --model $OUTPUT_PATH \
        --sub_task $EVAL_TASK \
        --output_file $OUTPUT_PATH/${EVAL_TASK}_response.jsonl

    python utils/test_acc.py \
        --input_file $OUTPUT_PATH/${EVAL_TASK}_response.jsonl
fi