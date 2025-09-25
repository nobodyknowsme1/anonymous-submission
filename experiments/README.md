# CoSA Experiments Guide

This directory contains experiment scripts for running various parameter-efficient fine-tuning methods on different models and tasks.

## Using the Common Runner

The `run_experiment.sh` script provides a unified way to run experiments across different methods and models.

### How to Run run_experiment.sh

```bash
# Run the experiment directly
bash experiments/run_experiment.sh [METHOD] [SEED] [MODEL]

# Example: Run LoRA with seed 42 and default model (llama31-8b)
bash experiments/run_experiment.sh lora 42

# Example: Run PiSSA with seed 123 and LLaMA-3.2-1B
bash experiments/run_experiment.sh pissa 123 llama32-1b

# Example: Run CoSA (compressed method) with seed 42 and Qwen2-7B
bash experiments/run_experiment.sh compressed 42 qwen2-7b

# Example: Use a custom model path with LoRA, seed 42
bash experiments/run_experiment.sh lora 42 meta-llama/custom-model
```

Arguments:
- `METHOD`: Training method (`lora`, `pissa`, `compressed`, `adalora`, `full`)
- `SEED`: Random seed, also serves as run ID (default: 42)
- `MODEL`: Model short name or full path (default: llama31-8b)

#### Predefined Model Shortcuts

| Shortcut | Full Model Path |
|----------|----------------|
| `llama31-8b` | `meta-llama/Llama-3.1-8B` (default) |
| `llama32-1b` | `meta-llama/Llama-3.2-1B` |
| `llama2-7b` | `meta-llama/Llama-2-7b-hf` |
| `qwen2-7b` | `Qwen/Qwen2-7B` |

The script will:
1. Parse command-line arguments (METHOD, SEED, MODEL)
2. Use SEED as both random seed and run ID for uniqueness
3. Resolve model shortcuts to full paths
4. Set up default configurations (can be overridden with environment variables)
5. Run training with DeepSpeed based on the selected method
6. If training succeeds, run evaluation using `utils/gen_vllm.py` and `utils/test_acc.py`
7. Save all results to the output directory

### Running Multiple Experiments

To run multiple experiments with different seeds, you can use a simple loop:

```bash
# Run LoRA 3 times with different seeds
for seed in 42 123 456; do
    bash experiments/run_experiment.sh lora $seed llama31-8b
done

# Run CoSA on Qwen2-7B with different seeds
for seed in 1 2 3; do
    bash experiments/run_experiment.sh compressed $seed qwen2-7b
done
```

### Customizing Experiments with Environment Variables

You can customize the experiments by setting environment variables before running the script:

```bash
# Example: Run LoRA with custom settings
BASE_MODEL="meta-llama/Llama-2-7b" LORA_RANK=256 BATCH_SIZE=8 bash experiments/run_experiment.sh lora 1 42

# Example: Run on specific GPUs
GPUS="0,1" bash experiments/run_experiment.sh pissa 1 42

# Example: Custom output path
OUTPUT_PATH="output/my-custom-experiment" bash experiments/run_experiment.sh compressed 42 llama31-8b
```

#### Core Configuration Variables

- `BASE_MODEL`: Model to fine-tune (overrides MODEL argument if set)
- `DATA_PATH`: Dataset location (default: "pissa-dataset", note: only test data included)
- `OUTPUT_PATH`: Output directory (default: auto-generated based on method and settings)
- `GPUS`: GPU IDs to use (default: "0,1,2,3")
- `NUM_EPOCHS`: Number of training epochs (default: 1)
- `MAX_LENGTH`: Maximum sequence length (default: 512)
- `BATCH_SIZE`: Per-device batch size (default: 2, method-specific)
- `GRAD_ACCUM`: Gradient accumulation steps (default: 8)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `SUB_TASK`: Dataset subset (default: "metamath:100000")
- `DATASET_FIELD`: Fields to use (default: "instruction output")
- `EVAL_TASK`: Evaluation task (default: "metamath")

### Method-Specific Parameters

#### LoRA
- `LORA_RANK`: Rank of adaptation (default: 128)
- `LORA_ALPHA`: Scaling factor (default: 128)
- `TARGET_MODULES`: Modules to adapt

#### PiSSA
- Same as LoRA parameters
- `RES_MODEL`: Path for residual model (auto-generated if not set)

#### Compressed
- `LORA_RANK`: Base rank (default: 128)
- `LORA_ALPHA`: Scaling factor (default: 128)
- `COMPRESSION_A`: First dimension (default: 1024)
- `COMPRESSION_B`: Second dimension (default: 256)

#### AdaLoRA
- `INIT_R`: Initial rank (default: 160)
- `ADALORA_A`: Alpha parameter (default: 128)
- `TARGET_R`: Target rank (default: 64)
- `ADALORA_LR`: Learning rate (default: 2e-4)

#### Full Fine-tuning
- `FULL_LR`: Learning rate (default: 1e-5)
- `FULL_WD`: Weight decay (default: 0.01)
- `FULL_WARMUP`: Warmup steps (default: 200)
- `ADAM_BETA1`: Adam beta1 (default: 0.9)
- `ADAM_BETA2`: Adam beta2 (default: 0.995)
- `ADAM_EPSILON`: Adam epsilon (default: 1e-8)
- `FULL_MAX_GRAD`: Max gradient norm (default: 0.5)
- `GRADIENT_CHECKPOINTING`: Enable gradient checkpointing (default: True)
- `FULL_LOG_STEPS`: Logging interval (default: 10)
- `FULL_EPOCHS`: Number of epochs (default: 3)
- `FULL_SAVE_STEPS`: Save interval (default: 1000)
- `FULL_BATCH_SIZE`: Batch size per device (default: 1)
- `FULL_GRAD_ACCUM`: Gradient accumulation steps (default: 8)

## Available Methods

1. **LoRA** (Low-Rank Adaptation)
   - Standard LoRA implementation with configurable rank and alpha
   - Default: rank=128, alpha=128

2. **PiSSA** (Principal Singular values and Singular vectors Adaptation)
   - Optimizes essential singular values/vectors while freezing noise
   - Requires initialization step before training
   - Uses fast SVD with 16 iterations

3. **Compressed** (CoSA: Compressed Sensing-based Adaptation)
   - Uses fixed random projections and compact trainable core
   - Novel L×Y×R factorization with compressed sensing theory
   - Default: a=1024, b=256

4. **AdaLoRA** (Adaptive Low-Rank Adaptation)
   - Dynamic rank allocation during training
   - Starts with initial rank and adapts to target rank
   - Default: init_r=160, target_r=64

5. **Full** (Full Fine-tuning)
   - Standard full model fine-tuning as baseline
   - Most resource-intensive method

## Supported Models and Tasks

### Mathematical Reasoning (MetaMath dataset)
- **LLaMA-3.1-8B**: `bash experiments/run_experiment.sh [method] [seed] llama31-8b`
- **LLaMA-3.2-1B**: `bash experiments/run_experiment.sh [method] [seed] llama32-1b`
- **LLaMA-2-7B**: `bash experiments/run_experiment.sh [method] [seed] llama2-7b`
- **Qwen2-7B**: `bash experiments/run_experiment.sh [method] [seed] qwen2-7b`

### Code Generation (Python dataset)
To run code generation experiments, set the task parameters:
```bash
SUB_TASK="python:50000" EVAL_TASK="python" bash experiments/run_experiment.sh [method] [seed] [model]
```

## Output Structure

Results are saved in the `output/` directory:
```
output/
├── metamath-lora-Llama-3.1-8B-r128-run1/
│   ├── adapter_model.safetensors
│   ├── config.json
│   ├── metamath_response.jsonl
│   └── trainer_state.json
├── PiSSA-Llama-3.1-8B-r128-run1/
│   └── (residual model files)
└── ...
```

## Common Configuration

All experiments use these default settings:
- **DeepSpeed**: Zero-2 optimization
- **Precision**: BF16
- **Epochs**: 1
- **Max sequence length**: 512
- **Gradient accumulation**: 8 steps
- **Learning rate**: 2e-5 (2e-4 for AdaLoRA)
- **Scheduler**: Cosine
- **Warmup ratio**: 0.03

## Evaluation

Evaluation runs automatically after training:
- **MetaMath**: Accuracy on mathematical reasoning
- **HumanEval/MBPP**: Pass@1 for code generation
- **GLUE**: Task-specific metrics (accuracy, F1, correlation)

Results are saved as:
- `metamath_response.jsonl` - Model responses
- `eval_results.json` - Computed metrics


## Creating Custom Experiments

### Example: New Model Integration

```bash
#!/bin/bash
# my_custom_experiment.sh

METHOD=$1
RUN_ID=$2
SEED=${3:-42}

# Set up environment variables for your custom model
export BASE_MODEL="your-model-id"
export DATA_PATH="pissa-dataset"
export OUTPUT_PATH="output/custom-${METHOD}-run${RUN_ID}"

# Optional: Override defaults
export BATCH_SIZE=4
export LEARNING_RATE=3e-5

# Run the experiment
bash experiments/run_experiment.sh "$METHOD" "$RUN_ID" "$SEED"

# Collect statistics
python utils/collect_results.py metamath "$METHOD" --model custom
```

## Environment Variables

Important paths that may need adjustment:
```bash
# HuggingFace cache locations
export HF_HOME=/your/path/to/huggingface
export TRANSFORMERS_CACHE=/your/path/to/transformers
export HF_DATASETS_CACHE=/your/path/to/datasets

# VLLM NCCL library (if using VLLM)
export VLLM_NCCL_SO_PATH="/path/to/libnccl.so.2"
```

## Advanced Usage

### Custom Hyperparameters

Set environment variables to customize hyperparameters:
```bash
# Custom LoRA configuration
LORA_RANK=256 LORA_ALPHA=512 TARGET_MODULES="q_proj,v_proj" \
bash experiments/run_experiment.sh lora 1 42
```

### Different GPU Configuration

```bash
# Use specific GPUs
GPUS="4,5,6,7" bash experiments/run_experiment.sh lora 1 42

# Or set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS="0,1,2,3" bash experiments/run_experiment.sh lora 1 42
```

### Custom Dataset

```bash
# Point to your dataset
DATA_PATH="my_custom_dataset" SUB_TASK="custom_task:50000" DATASET_FIELD="input output" \
bash experiments/run_experiment.sh lora 1 42
```

## Best Practices

1. **Always run multiple seeds** (3+ recommended) for reliable results
2. **Monitor GPU usage** with `nvidia-smi -l 1` during training
3. **Check logs** in output directories for debugging
4. **Use the common runner** for new experiments to maintain consistency
5. **Document changes** when modifying default parameters