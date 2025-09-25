import copy
import json
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import os
import time
import numpy as np

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig, TrainerState, TrainerControl, EarlyStoppingCallback
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, concatenate_datasets
import datasets
from peft import LoraConfig, AdaLoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

# Configure logging to show INFO messages
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

# GLUE task configurations
GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"), 
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

GLUE_TASK_TO_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "stsb": 1,  # regression task
    "wnli": 2,
}

@dataclass
class GLUETrainingArguments(transformers.TrainingArguments):
    # Base model setting
    model_name_or_path: Optional[str] = field(default="roberta-base")
    
    # PEFT settings
    full_finetune: Optional[bool] = field(default=False)
    adapter_name_or_path: Optional[str] = field(default=None)
    init_weights: str = field(default="True", metadata={"help": "True -> LoRA; `pissa` -> PiSSA; `pissa_niter_16` -> Fast SVD PiSSA"})
    use_dora: Optional[bool] = field(default=False)
    target_modules: Optional[str] = field(default="attention.self.query,attention.self.key,attention.self.value,attention.output.dense")
    lora_rank: Optional[int] = field(default=8)
    lora_alpha: Optional[float] = field(default=32.)
    lora_dropout: Optional[float] = field(default=0.1)
    
    # Quantization setting
    bits: int = field(default=16)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    
    # GLUE specific settings
    task_name: str = field(default="sst2", metadata={"help": "GLUE task name"})
    max_seq_length: int = field(default=128, metadata={"help": "Maximum sequence length"})
    merge: bool = field(default=True, metadata={"help": "Merge adapter weights after training"})
    early_stopping_patience: int = field(default=3, metadata={"help": "Early stopping patience (RoBERTa paper recommendation)"})
    
    # AdaLoRA specific settings
    use_adalora: bool = field(default=False, metadata={"help": "Use AdaLoRA instead of LoRA"})
    adalora_init_r: int = field(default=12, metadata={"help": "Initial rank for AdaLoRA"})
    adalora_a: int = field(default=32, metadata={"help": "Alpha parameter for AdaLoRA"})
    adalora_target_r: int = field(default=8, metadata={"help": "Target rank for AdaLoRA"})

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        if "tokenizer" in kwargs:
            kwargs["tokenizer"].save_pretrained(checkpoint_folder)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        if "tokenizer" in kwargs:
            kwargs["tokenizer"].save_pretrained(args.output_dir)

class LossTrackerCallback(transformers.TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
        self.train_losses = []
        self.train_steps = []
        self.train_times = []
        self.start_time = time.time()

    def on_log(self, args: transformers.TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if "loss" in logs:
            elapsed_time = time.time() - self.start_time
            step = state.global_step
            loss = logs["loss"]

            self.train_losses.append(loss)
            self.train_steps.append(step)
            self.train_times.append(elapsed_time)

            # self.logger.info(f"[Step {step}] Loss: {loss:.4f}, Elapsed Time: {elapsed_time:.2f} seconds")

    def get_tracking_data(self):
        return {
            "steps": self.train_steps,
            "losses": self.train_losses,
            "times": self.train_times
        }

def compute_metrics(eval_pred, task_name):
    predictions, labels = eval_pred

    if task_name == "stsb":
        # Regression task
        predictions = predictions[:, 0]
        pearson_corr = pearsonr(predictions, labels)[0]
        spearman_corr = spearmanr(predictions, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "combined_score": (pearson_corr + spearman_corr) / 2
        }
    else:
        # Classification tasks
        predictions = np.argmax(predictions, axis=1)

        if task_name == "cola":
            matthews_corr = matthews_corrcoef(labels, predictions)
            return {
                "matthews_correlation": matthews_corr,
                "accuracy": accuracy_score(labels, predictions)
            }
        elif task_name in ["mrpc", "qqp"]:
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions)
            }
        else:
            return {"accuracy": accuracy_score(labels, predictions)}

def preprocess_function(examples, tokenizer, task_name, max_seq_length):
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]

    if sentence2_key is None:
        # Single sentence tasks
        texts = examples[sentence1_key]
        encoding = tokenizer(texts, truncation=True, padding=True, max_length=max_seq_length)
    else:
        # Sentence pair tasks
        texts = list(zip(examples[sentence1_key], examples[sentence2_key]))
        encoding = tokenizer(*zip(*texts), truncation=True, padding=True, max_length=max_seq_length)

    # Add labels to the encoding
    if "label" in examples:
        encoding["labels"] = examples["label"]

    return encoding

def build_model(script_args):
    compute_dtype = torch.bfloat16 if script_args.bf16 else torch.float32
    
    num_labels = GLUE_TASK_TO_LABELS[script_args.task_name]
    
    if script_args.bits != 16:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name_or_path,
            num_labels=num_labels,
            quantization_config=bnb_config,
            attn_implementation=getattr(script_args, 'attn_implementation', None),
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            device_map={"": int(os.environ.get("LOCAL_RANK", 0))}
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name_or_path,
            num_labels=num_labels,
            attn_implementation=getattr(script_args, 'attn_implementation', None),
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            device_map={"": int(os.environ.get("LOCAL_RANK", 0))}
        )

    if not script_args.full_finetune:
        if script_args.bits != 16:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

        target_modules = script_args.target_modules.split(',')
        
        # Convert init_weights string to appropriate type
        if script_args.init_weights.lower() == "true":
            init_lora_weights = True
        elif script_args.init_weights.lower() == "false":
            init_lora_weights = False
        else:
            init_lora_weights = script_args.init_weights  # Keep as string for PiSSA values
        
        if script_args.use_adalora:
            lora_config = AdaLoraConfig(
                init_r=script_args.adalora_init_r,
                target_r=script_args.adalora_target_r,
                r=script_args.adalora_init_r,
                lora_alpha=script_args.adalora_a,
                target_modules=target_modules,
                lora_dropout=script_args.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
        else:
            lora_config = LoraConfig(
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=script_args.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS,
                init_lora_weights=init_lora_weights,
                use_dora=script_args.use_dora,
            )
        
        if script_args.adapter_name_or_path is not None:
            model = PeftModel.from_pretrained(model, script_args.adapter_name_or_path, is_trainable=True)
            logger.info(f"Loaded adapter from {script_args.adapter_name_or_path}")
        else:
            model = get_peft_model(model, lora_config)
            logger.info("Created new PEFT model")

        model.print_trainable_parameters()

    return model

def get_last_checkpoint(checkpoint_dir):
    # Always start fresh to avoid optimizer state mismatch issues
    # This prevents incompatible checkpoint resumption when model configs change
    return None

def train():
    parser = transformers.HfArgumentParser(GLUETrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    log_level = script_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
        
    if script_args.local_rank == 0:
        logger.info('='*100)
        logger.info(script_args)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True
    )
    
    # Load dataset
    if script_args.task_name == "mnli":
        raw_datasets = load_dataset("glue", script_args.task_name)
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation_matched"]
    else:
        raw_datasets = load_dataset("glue", script_args.task_name)
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation"]

    if script_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(script_args.model_name_or_path))
        logger.info(f"Training on task: {script_args.task_name}")
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Build model
    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    model = build_model(script_args)

    # Preprocess datasets
    def tokenize_function(examples):
        return preprocess_function(examples, tokenizer, script_args.task_name, script_args.max_seq_length)

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on train dataset",
    )

    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Running tokenizer on validation dataset",
    )

    # Debug: Check if evaluation dataset has labels
    if script_args.local_rank == 0:
        logger.info(f"Eval dataset columns after preprocessing: {eval_dataset.column_names}")
        logger.info(f"Eval dataset features: {eval_dataset.features}")
        if len(eval_dataset) > 0:
            sample = eval_dataset[0]
            logger.info(f"Sample from eval dataset: {sample}")
            # Check if labels key exists
            if 'labels' in sample:
                logger.info(f"Labels found in sample: {sample['labels']}")
            else:
                logger.warning("WARNING: No 'labels' key found in eval dataset sample!")

    # Set up compute_metrics function
    def compute_metrics_fn(eval_pred):
        result = compute_metrics(eval_pred, script_args.task_name)
        if script_args.local_rank == 0:
            logger.info(f"Compute metrics called for {script_args.task_name}, returning: {result}")
        return result

    # Set up trainer
    trainer = Trainer(
        model=model,
        args=script_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # Use tokenizer parameter for compatibility
        compute_metrics=compute_metrics_fn,
    )

    # Add callbacks
    if not script_args.full_finetune:
        trainer.add_callback(SavePeftModelCallback())
    
    loss_tracker = LossTrackerCallback(logger=logger)
    trainer.add_callback(loss_tracker)
    
    # Add early stopping (RoBERTa paper uses early stopping)
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=script_args.early_stopping_patience))

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint_dir)
    trainer.save_state()

    # Save model
    if not script_args.full_finetune and script_args.merge:
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
    elif script_args.full_finetune:
        trainer.save_model()
        
    # Save loss tracking
    loss_data = loss_tracker.get_tracking_data()
    loss_file = os.path.join(script_args.output_dir, "loss_tracking.json")
    with open(loss_file, "w") as f:
        json.dump(loss_data, f, indent=2)

    logger.info(f"Saved training loss tracking to {loss_file}")

    # Final evaluation
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")

    # Save evaluation results
    eval_file = os.path.join(script_args.output_dir, "eval_results.json")
    with open(eval_file, "w") as f:
        json.dump(eval_results, f, indent=2)

if __name__ == "__main__":
    train()