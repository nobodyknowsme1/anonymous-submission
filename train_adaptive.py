import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import os
import time
import sys

# Add the local peft path 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'peft/src'))

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig, TrainerState, TrainerControl, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
import datasets
import numpy as np
import peft
print(f"Using PEFT from: {peft.__file__}")
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel, LoraRuntimeConfig, CoSAConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

# Configure logging to show INFO messages
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Base model or residual model setting
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    # Lora or PiSSA setting
    full_finetune : Optional[bool] = field(default=True)
    adapter_name_or_path: Optional[str] = field(default=None,metadata={"help": ("Pre-initialized PiSSA adapter path; when this is not None, the following arguments are ignored."),},)
    init_weights: str = field(default="True",metadata={"help": ("True -> LoRA; `pissa` -> PiSSA; `pissa_niter_16` -> Fast SVD PiSSA"),},)
    use_dora : Optional[bool] = field(default=False)
    target_modules : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_alpha : Optional[float] = field(default=32.)
    lora_dropout : Optional[float] = field(default=0.,metadata={"help": ("Must be set to 0 when using PiSSA."),},)
    
    # Adaptive compression parameters
    use_adaptive_compression: Optional[bool] = field(default=True, metadata={"help": "Enable adaptive layer-wise compression"})
    compression_a: Optional[int] = field(default=1024, metadata={"help": "Default compression parameter a for SumLoRA (used when adaptive is off)"})
    compression_b: Optional[int] = field(default=256, metadata={"help": "Default compression parameter b for SumLoRA (used when adaptive is off)"})
    
    # Quantization setting
    bits: int = field(default=16,metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True,metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    # DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    sub_task: List[str] = field(default=None)
    dataset_split: str = field(default="train", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default_factory=lambda:["instruction", "output"], metadata={"help": "Fields to use from the dataset."})
    # TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    use_flash_attention_2: bool = field(default=False,metadata={"help": "Whether to use flash attention 2."})
    merge : Optional[bool] = field(default=False)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

@torch.no_grad()
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    # Load raw datasets
    if ":" in data_args.sub_task[0]:
        cur_task, num_split = data_args.sub_task[0].split(":")
        cur_split = f"{data_args.dataset_split}[:{num_split}]"
    else:
        cur_task, cur_split = data_args.sub_task[0], data_args.dataset_split

    raw_train_datasets = load_dataset(data_args.data_path, data_dir=cur_task, split=cur_split)
    
    logger.info(f"Total training samples before filtering: {len(raw_train_datasets)}")
    
    # Apply filtering if needed (similar to train_compressed.py pattern)
    if any("math" in task.lower() for task in data_args.sub_task):
        # For math tasks, filter for math/gsm8k data if type field exists
        if "type" in raw_train_datasets.column_names:
            raw_train_datasets = raw_train_datasets.filter(lambda x: x["type"].lower().startswith("math") or x["type"].lower().startswith("gsm"))
        logger.info(f"Total training samples after math filtering: {len(raw_train_datasets)}")
    else:
        logger.info(f"Total training samples after filtering (no filtering applied): {len(raw_train_datasets)}")

    # Shuffle dataset with seed if available
    if hasattr(data_args, 'seed') and data_args.seed:
        logger.info(f"Shuffle dataset with seed={data_args.seed}")
        raw_train_datasets = raw_train_datasets.shuffle(seed=data_args.seed)
    else:
        # Use default seed for reproducibility
        logger.info("Shuffle dataset with default seed=42")
        raw_train_datasets = raw_train_datasets.shuffle(seed=42)
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": data_args.dataset_field[0], "response": data_args.dataset_field[1]}
    )

    logger.info(f"Training dataset samples: {len(train_dataset)}")
    if len(train_dataset) > 0:
        sample_size = min(3, len(train_dataset))
        for index in random.sample(range(len(train_dataset)), sample_size):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            logger.info(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")
    else:
        logger.error("Training dataset is empty! Please check your data filtering logic.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class SavePeftModelCallback(transformers.TrainerCallback):
    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        logs=None,
        **kwargs,
    ):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "sft_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}", "sft_lora_model")

        peft_model_path = os.path.join(checkpoint_folder)
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

        return control

class LossTrackerCallback(transformers.TrainerCallback):
    def __init__(self):
        self.training_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "train_loss" in logs:
            self.training_losses.append({
                "step": state.global_step,
                "loss": logs["train_loss"],
                "epoch": state.epoch
            })
            
    def on_train_end(self, args, state, control, **kwargs):
        # Save loss tracking to output directory
        loss_file = os.path.join(args.output_dir, "loss_tracking.json")
        import json
        with open(loss_file, 'w') as f:
            json.dump(self.training_losses, f, indent=2)
        print(f"Loss tracking saved to {loss_file}")

def get_model_and_tokenizer(script_args, model_args):
    # Load quantization config
    if script_args.bits in [4, 8]:
        compute_dtype = getattr(torch, script_args.dtype) if hasattr(script_args, 'dtype') else torch.bfloat16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        )
    else:
        quant_config = None

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=script_args.cache_dir,
        quantization_config=quant_config,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch.bfloat16 if script_args.bf16 else None,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=script_args.cache_dir,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # Set pad token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="<pad>"),
            tokenizer=tokenizer,
            model=model,
        )

    return model, tokenizer

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def train():
    parser = transformers.HfArgumentParser((TrainingArguments,))
    script_args, = parser.parse_args_into_dataclasses()
    
    print("=== Adaptive Compression SumLoRA Training ===")
    print(f"Model: {script_args.model_name_or_path}")
    print(f"Adaptive compression: {script_args.use_adaptive_compression}")
    if not script_args.use_adaptive_compression:
        print(f"Uniform compression: a={script_args.compression_a}, b={script_args.compression_b}")

    # Get model and tokenizer
    model, tokenizer = get_model_and_tokenizer(script_args, script_args)

    # Configure PEFT
    if not script_args.full_finetune:
        if script_args.bits in [4, 8]:
            model = prepare_model_for_kbit_training(model)

        # Convert init_weights string to appropriate type
        if script_args.init_weights.lower() in ['true', '1', 'yes', 't', 'y']:
            init_weights_val = True
        elif script_args.init_weights.lower() in ['false', '0', 'no', 'f', 'n']:
            init_weights_val = False
        else:
            # For PiSSA initialization strings like 'pissa', 'pissa_niter_16'
            init_weights_val = script_args.init_weights

        # Setup SumLoRA config with adaptive compression
        config = CoSAConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=script_args.lora_rank,
            lora_alpha=int(script_args.lora_alpha),
            lora_dropout=script_args.lora_dropout,
            target_modules=script_args.target_modules.split(","),
            bias="none",
            use_rslora=False,
            modules_to_save=None,
            init_lora_weights=init_weights_val,
            use_dora=script_args.use_dora,
            runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False),
            normalize_sum=False,
            use_sum_lora=True,
            
            use_compression=True,
            use_adaptive_compression=script_args.use_adaptive_compression,  # 🔥 Key parameter
            compression_a=script_args.compression_a,
            compression_b=script_args.compression_b,
        )

        print(f"SumLoRA Config: {config}")
        
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        
        # Print per-layer compression info
        if script_args.use_adaptive_compression:
            print("\n=== Adaptive Layer-wise Compression Configuration ===")
            layer_count = 0
            total_params = 0
            for name, module in model.named_modules():
                if hasattr(module, 'compression_a') and hasattr(module, 'compression_b'):
                    for adapter_name in getattr(module, 'compression_a', {}):
                        comp_a = module.compression_a[adapter_name]
                        comp_b = module.compression_b[adapter_name]
                        if hasattr(module, 'base_layer'):
                            in_features = getattr(module.base_layer, 'in_features', 'N/A')
                            out_features = getattr(module.base_layer, 'out_features', 'N/A')
                            if isinstance(in_features, int) and isinstance(out_features, int):
                                layer_params = comp_a * comp_b + out_features * comp_a + comp_b * in_features
                                total_params += layer_params
                                print(f"  {name}: ({in_features},{out_features}) -> a={comp_a},b={comp_b} -> {layer_params:,} params")
                                layer_count += 1
            print(f"Total adaptive layers: {layer_count}, Total compression params: {total_params:,}")
        
    # Setup data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=script_args)

    # Setup trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=script_args,
        **data_module
    )

    # Add callbacks
    trainer.add_callback(SavePeftModelCallback)
    trainer.add_callback(LossTrackerCallback())

    # Train the model
    trainer.train()
    trainer.save_state()

    # Merge and save if requested
    if script_args.merge and not script_args.full_finetune:
        output_dir = os.path.join(script_args.output_dir, "merged_model")
        model.merge_and_unload().save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Merged model saved to {output_dir}")
    else:
        model.save_pretrained(script_args.output_dir)

if __name__ == "__main__":
    train()