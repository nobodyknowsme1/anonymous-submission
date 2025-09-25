# **Co**mpressed **S**ensing-based **A**daptation (CoSA)

## Abstract

Parameter-Efficient Fine-Tuning (PEFT) has emerged as a practical paradigm for adapting large language models (LLMs) without updating all parameters. Most existing approaches, such as LoRA and PiSSA, rely on low-rank decompositions of weight updates. However, the low-rank assumption may restrict expressivity, particularly in task-specific adaptation scenarios where singular values are distributed relatively uniformly.

To address this limitation, we propose **CoSA** (*Compressed Sensing-Based Adaptation*), a new PEFT method extended from compressed sensing theory. Instead of constraining weight updates to a low-rank subspace, CoSA expresses them through fixed random projection matrices and a compact learnable core. We provide a formal theoretical analysis of CoSA as a synthesis process, proving that weight updates can be compactly encoded into a low-dimensional space and mapped back through random projections.

Extensive experimental results suggest that CoSA provides a principled perspective for efficient and expressive multi-scale model adaptation. Specifically, we evaluate CoSA on 10 diverse tasks including natural language understanding and generation, employing 5 models of different scales from RoBERTa, Llama, and Qwen families. Across these settings, CoSA consistently matches or outperforms state-of-the-art PEFT baselines while requiring over 68.4% fewer trainable parameters than LoRA and PiSSA.

## Key Contributions

- We propose a compressed sensing–based PEFT method with fixed random projections and a compact trainable core from a fundamentally different perspective compared to LoRA.
- A theoretical foundation is provided by framing CoSA as a synthesis process in compressed sensing, proving that its Kronecker dictionary of random projections satisfies the Restricted Isometry Property (RIP), ensuring near-isometry and stable optimization.
- Extensive experiments on NLU and NLG benchmarks with RoBERTa, LLaMA, and Qwen show that CoSA matches or outperforms state-of-the-art PEFT methods while offering substantial parameter savings.

## Quick Start

### Environment Setup
```bash
# Create conda environment
conda create -n cosa python=3.10
conda activate cosa

# Install CUDA toolkit.
conda install nvidia/label/cuda-12.1.0::cuda-toolkit

# Install PyTorch with CUDA support
conda install pytorch==2.4.0 torchvision=0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install NCCL for multi-GPU support (REQUIRED for VLLM multi-GPU inference)
conda install -c nvidia nccl=2.28.3

# Install dependencies
pip install -r requirements.txt

# Install custom PEFT with CoSA features (editable)
pip install -e peft

# Install Flash Attention (optional, for better performance)
pip install flash-attn --no-build-isolation

# Note: Only test datasets are included in this repo for evaluation
# For full training datasets, download from HuggingFace:
pip install -U huggingface_hub
huggingface-cli download --repo-type dataset --resume-download [ANONYMOUS]/pissa-dataset --local-dir pissa-dataset
```

### Verify Installation
```bash
python -c "import torch, transformers, peft, datasets; print('✓ Installation successful!')"
python -c "import torch; print('NCCL available:', torch.distributed.is_nccl_available())"
```

## Running Experiments

All experiment instructions and configurations are provided in the `experiments/` directory. The repository includes a unified experiment runner that supports various PEFT methods including CoSA.

### Quick Experiment Examples

```bash
# Run CoSA (compressed method) with LLaMA-3.1-8B
bash experiments/run_experiment.sh compressed 42 llama31-8b

# Run CoSA with LLaMA-3.2-1B
bash experiments/run_experiment.sh compressed 123 llama32-1b

# Run CoSA with Qwen2-7B
bash experiments/run_experiment.sh compressed 456 qwen2-7b

# Compare with LoRA baseline
bash experiments/run_experiment.sh lora 42 llama31-8b

# Compare with PiSSA baseline
bash experiments/run_experiment.sh pissa 42 llama31-8b
```

### Multiple Runs for Statistical Significance
```bash
# Run CoSA with multiple seeds for robust evaluation
for seed in 1 2 3; do
    bash experiments/run_experiment.sh compressed $seed llama31-8b
done
```

**For detailed experiment instructions, configurations, and advanced usage, please see [experiments/README.md](experiments/README.md).**

## Supported Models and Tasks

### Mathematical Reasoning (MetaMath dataset)
- **LLaMA-3.1-8B**: High-performance mathematical reasoning
- **LLaMA-3.2-1B**: Efficient small-scale model
- **LLaMA-2-7B**: Standard benchmark model
- **Qwen2-7B**: Alternative architecture evaluation 

### Natural Language Understanding (GLUE benchmark)
- **RoBERTa-base/large**: Comprehensive NLU evaluation
- Multiple tasks: CoLA, SST-2, MRPC, STS-B, QNLI, RTE

### Code Generation (Python dataset)
- **HumanEval/MBPP**: Code synthesis evaluation
- Pass@1 metrics for programming tasks

## CoSA Architecture

CoSA uses a novel **L×Y×R factorization** where:
- **L matrix**: Fixed random projection (frozen, deterministically generated)
- **Y matrix**: Trainable adaptation layer (compact core)
- **R matrix**: Fixed random projection (frozen, deterministically generated)
- **Forward pass**: `result += L(Y(R(x))) * scaling`

This architecture provides structured compression while maintaining adaptation capability with significantly fewer parameters than traditional low-rank methods.

## Evaluation

The repository includes automated evaluation pipelines:
- **MetaMath**: Accuracy on mathematical reasoning
- **HumanEval/MBPP**: Pass@1 for code generation
- **GLUE**: Task-specific metrics (accuracy, F1, correlation)

Results are automatically saved after training completion.

## Advanced Usage

### Using CoSA in Custom Code

```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Configure CoSA (compressed method)
lora_config = LoraConfig(
    init_lora_weights="compressed",  # Use CoSA initialization
    r=128,
    lora_alpha=128,
    lora_dropout=0,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```


### Quick NCCL Fix
If you encounter `'PyNcclCommunicator' object has no attribute 'nccl'` errors:

```bash
# Install NCCL
conda install -c nvidia nccl=2.28.3

# Set environment variable
export VLLM_NCCL_SO_PATH="$CONDA_PREFIX/lib/libnccl.so.2"

# Verify installation
python -c "import torch; print('NCCL available:', torch.distributed.is_nccl_available())"
```

**Note**: This repository contains the implementation and experimental setup for CoSA, a novel compressed sensing-based approach to parameter-efficient fine-tuning.# anonymous-submission
