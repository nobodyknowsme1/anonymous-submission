import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import os
import time

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig, TrainerState, TrainerControl, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
import datasets
import numpy as np
import peft
print(peft.__file__)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel, LoraRuntimeConfig, CoSAConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from utils.gpu_memory_monitor import GPUMemoryMonitor, MemoryMonitorCallback

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
    init_weights: bool | str = field(default=True,metadata={"help": ("True -> LoRA; `pissa` -> PiSSA; `pissa_niter_16` -> Fast SVD PiSSA"),},)
    use_dora : Optional[bool] = field(default=False)
    target_modules : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_alpha : Optional[float] = field(default=32.)
    lora_dropout : Optional[float] = field(default=0.,metadata={"help": ("Must be set to 0 when using PiSSA."),},)
    # Compression parameters
    compression_a: Optional[int] = field(default=1024, metadata={"help": "Compression parameter a for CoSA"})
    compression_b: Optional[int] = field(default=256, metadata={"help": "Compression parameter b for CoSA"})
    # Quantization setting
    bits: int = field(default=16,metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True,metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    # DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    sub_task: List[str] = field(default=None)
    dataset_split: str = field(default="train", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    shuffle_dataset : Optional[bool] = field(default=False)
    validation_split_percentage: Optional[int] = field(default=0, metadata={"help": "Percentage of dataset to use for validation."})
    # TrainingArguments
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    merge : Optional[bool] = field(default=False,metadata={"help": "Merge the PiSSA adapter to the residual model or LoRA to the base model"},)

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

class LossTrackerCallback(transformers.TrainerCallback):
    def __init__(self, logger: logging.Logger = None):
        self.train_losses = []
        self.train_steps = []
        self.train_times = []
        self.start_time = None

        # Setup logger
        self.logger = logger or logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()
        self.logger.info("Training started.")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
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


def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None # first training

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,truncation=True,)for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

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

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def build_model(script_args, checkpoint_dir):
    if script_args.full_finetune:
        assert script_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if script_args.bf16 else torch.float32)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        ) if script_args.bits in [4, 8] else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    # Tokenizer
    
    if not script_args.full_finetune:
        if script_args.bits < 16:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            # os.path.join(checkpoint_dir, 'adapter_model')
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        elif script_args.adapter_name_or_path is not None:
            logger.info(f"Initilize LoRA/PiSSA/CLOVER adapters from {script_args.model_name_or_path}/{script_args.adapter_name_or_path}.")
            model = PeftModel.from_pretrained(model, script_args.model_name_or_path, subfolder = script_args.adapter_name_or_path, is_trainable=True)
        else:
            logger.info(f'Init LoRA/PiSSA modules...')
            peft_config = LoraConfig(
                use_dora=script_args.use_dora,
                runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=script_args.use_dora),
                task_type=TaskType.CAUSAL_LM,
                target_modules=script_args.target_modules.split(','),
                inference_mode=False,
                r=script_args.lora_rank, 
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                init_lora_weights=script_args.init_weights,
            )
            model = get_peft_model(model, peft_config)

    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
    return model

def build_own_model(script_args, checkpoint_dir, adapter_mode="compressed"):
    if adapter_mode == "pissa":
        return build_model(script_args, checkpoint_dir)
    elif adapter_mode == "lora":
        return build_model(script_args, checkpoint_dir)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path, 
        use_auth_token=True,
        # device_map="auto",
    )

    if checkpoint_dir is not None:
        print(f"Loading adapters from {checkpoint_dir}.")
        # os.path.join(checkpoint_dir, 'adapter_model')
        model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=True)
    else:
        print(f"Initilize adapters from {script_args.model_name_or_path}.")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        compression_config = CoSAConfig(
            r=script_args.lora_rank,
            lora_alpha=script_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights = False,
            use_compression = True,
            compression_a=script_args.compression_a,
            compression_b=script_args.compression_b,
        )
        model = get_peft_model(base_model, compression_config)

    # Inject experiment seed for on-demand matrix generation
    if hasattr(script_args, 'seed') and script_args.seed is not None:
        for module in model.modules():
            if hasattr(module, 'set_experiment_seed'):
                module.set_experiment_seed(script_args.seed)
                print(f"Set experiment seed {script_args.seed} for compression matrix generation")

    model.print_trainable_parameters()
    return model

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
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
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(script_args.model_name_or_path))
    
    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    print(f"Resume from checkpoint: {resume_from_checkpoint_dir}")
    # model = build_model(script_args, resume_from_checkpoint_dir)
    model = build_own_model(script_args, resume_from_checkpoint_dir, adapter_mode="compressed")

    all_training_dataset = []
    for task in script_args.sub_task:
        if ":" in task: # e.g. math:500, gsm8k:100
            cur_task, num_split = task.split(":")
            cur_split = f"{script_args.dataset_split}[:{num_split}]"
        else:
            cur_task, cur_split = task, script_args.dataset_split

        ds = load_dataset(script_args.data_path, data_dir=cur_task, split=cur_split)
        if script_args.local_rank == 0:
            print(f"{script_args.data_path}/{cur_task}/{cur_split}/{ds.num_rows}")
            for k,v in ds[0].items():
                print("-"*100)
                print(k,end=':\t')
                print(v)
            print("+"*100)
        all_training_dataset.append(ds)
        
    raw_train_datasets = concatenate_datasets(all_training_dataset)

    # Apply task-specific filtering based on subtask
    logger.info(f"Total training samples before filtering: {len(raw_train_datasets)}")
    
    # Determine filter based on subtask
    if any("math" in task.lower() for task in script_args.sub_task):
        # For math tasks, filter for math/gsm8k data
        raw_train_datasets = raw_train_datasets.filter(lambda x: x["type"].lower().startswith("math") or x["type"].lower().startswith("gsm"))
        logger.info(f"Total training samples after math filtering: {len(raw_train_datasets)}")
    elif any("python" in task.lower() for task in script_args.sub_task):
        # For python tasks, don't filter or filter for code-related data
        logger.info(f"Total training samples after python filtering (no filtering applied): {len(raw_train_datasets)}")
    else:
        # For other tasks, keep all data
        logger.info(f"Total training samples after filtering (no filtering applied): {len(raw_train_datasets)}")

    if script_args.shuffle_dataset:
        if script_args.local_rank == 0:
            logger.info(f"Shuffle dataset with seed={script_args.seed}")
        raw_train_datasets = raw_train_datasets.shuffle(seed=script_args.seed)

    if script_args.local_rank > 0: 
        torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0], "response": script_args.dataset_field[1]}
    )

    if script_args.local_rank == 0:
        torch.distributed.barrier()
        logger.info("Training dataset samples:", len(train_dataset))
        if len(train_dataset) > 0:
            sample_size = min(3, len(train_dataset))
            for index in random.sample(range(len(train_dataset)), sample_size):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
                logger.info(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")
        else:
            logger.error("Training dataset is empty! Please check your data filtering logic.")

    # Check if we have training data
    if len(train_dataset) == 0:
        logger.error("No training data available after filtering. Exiting.")
        return

    # Create validation split if specified
    eval_dataset = None
    if script_args.validation_split_percentage > 0:
        total_size = len(train_dataset)
        eval_size = int(total_size * script_args.validation_split_percentage / 100)
        train_size = total_size - eval_size
        
        train_dataset, eval_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, eval_size]
        )
        
        if script_args.local_rank == 0:
            logger.info(f"Split dataset: {train_size} train, {eval_size} eval")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    if not script_args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    loss_tracker = LossTrackerCallback(logger=logger)
    trainer.add_callback(loss_tracker)

    # Add GPU memory monitoring
    memory_monitor = GPUMemoryMonitor(output_dir=script_args.output_dir, log_interval=10)
    memory_callback = MemoryMonitorCallback(memory_monitor)
    trainer.add_callback(memory_callback)
    trainer.train(resume_from_checkpoint = resume_from_checkpoint_dir)
    # trainer.train()
    trainer.save_state()
    if not script_args.full_finetune and script_args.merge:        
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
    if script_args.full_finetune:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)
        
    import json
    loss_data = loss_tracker.get_tracking_data()
    loss_file = os.path.join(script_args.output_dir, "loss_tracking.json")
    with open(loss_file, "w") as f:
        json.dump(loss_data, f, indent=2)

    logger.info(f"Saved training loss tracking to {loss_file}")

if __name__ == "__main__":
    train()
