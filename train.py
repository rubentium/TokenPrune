"""
Training script for Qwen3 with multi-GPU support, wandb logging, and flash attention.

Usage:
    # Single GPU
    python train.py --config train_config.json
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 train.py --config train_config.json
    
    # Multi-GPU with accelerate
    accelerate launch train.py --config train_config.json
"""

import argparse
import json
import os
import math
import sys
import time
import wandb
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
from dataclasses import dataclass
from datasets import load_from_disk

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from config import Qwen3Config
from model import Qwen3ForCausalLM, load_tokenizer

try: 
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

try:
    from torchinfo import summary
except ImportError:
    summary = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TrainConfig:
    """Training configuration loaded from JSON."""
    # Model
    model_path: str = "../qwen3_06b_base"
    model_dtype: str = "bfloat16"
    init_from_scratch: bool = False
    
    # Data
    train_path: str = "./data/sample/sample.jsonl"
    val_path: Optional[str] = None
    max_seq_length: int = 512
    num_workers: int = 8
    
    # Training
    output_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint dir to resume from
    num_epochs: int = 1
    max_steps: Optional[int] = None
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    eval_batches: int = 100
    logging_steps: int = 10
    seed: int = 42
    verbose: bool = False
    checkpoint_activations: bool = False
    
    # Distributed
    strategy: str = "ddp"
    mixed_precision: str = "bf16"
    
    # Wandb
    wandb_enabled: bool = True
    wandb_project: str = "qwen3-training"
    wandb_name: Optional[str] = None
    wandb_tags: Union[list, None] = None
    
    # Flash attention
    flash_attention: bool = True
    
    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            data = json.load(f)
        
        config = cls()
        
        # Flatten nested config
        if "model" in data:
            config.model_path = data["model"].get("path", config.model_path)
            config.model_dtype = data["model"].get("dtype", config.model_dtype)
            config.init_from_scratch = data["model"].get("init_from_scratch", config.init_from_scratch)
        
        if "data" in data:
            config.train_path = data["data"].get("train_path", config.train_path)
            config.val_path = data["data"].get("val_path", config.val_path)
            config.max_seq_length = data["data"].get("max_seq_length", config.max_seq_length)
            config.num_workers = data["data"].get("num_workers", config.num_workers)
        
        if "training" in data:
            t = data["training"]
            config.output_dir = t.get("output_dir", config.output_dir)
            config.resume_from_checkpoint = t.get("resume_from_checkpoint", config.resume_from_checkpoint)
            config.num_epochs = t.get("num_epochs", config.num_epochs)
            config.max_steps = t.get("max_steps", config.max_steps)
            config.per_device_batch_size = t.get("per_device_batch_size", config.per_device_batch_size)
            config.gradient_accumulation_steps = t.get("gradient_accumulation_steps", config.gradient_accumulation_steps)
            config.learning_rate = t.get("learning_rate", config.learning_rate)
            config.weight_decay = t.get("weight_decay", config.weight_decay)
            config.warmup_steps = t.get("warmup_steps", config.warmup_steps)
            config.lr_scheduler = t.get("lr_scheduler", config.lr_scheduler)
            config.max_grad_norm = t.get("max_grad_norm", config.max_grad_norm)
            config.save_steps = t.get("save_steps", config.save_steps)
            config.eval_steps = t.get("eval_steps", config.eval_steps)
            config.eval_batches = t.get("eval_batches", config.eval_batches)
            config.logging_steps = t.get("logging_steps", config.logging_steps)
            config.seed = t.get("seed", config.seed)
            config.verbose = t.get("verbose", config.verbose)
            config.checkpoint_activations = t.get("checkpoint_activations", config.checkpoint_activations)
        if "distributed" in data:
            config.strategy = data["distributed"].get("strategy", config.strategy)
            config.mixed_precision = data["distributed"].get("mixed_precision", config.mixed_precision)
        
        if "wandb" in data:
            config.wandb_enabled = data["wandb"].get("enabled", config.wandb_enabled)
            config.wandb_project = data["wandb"].get("project", config.wandb_project)
            config.wandb_name = data["wandb"].get("name", config.wandb_name)
            config.wandb_tags = data["wandb"].get("tags", config.wandb_tags)
        
        config.flash_attention = data.get("flash_attention", config.flash_attention)
        
        return config


class TextDataset(Dataset):
    """Simple text dataset for language model training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []
        
        # Load data
        if data_path.endswith(".jsonl"):
            self._load_jsonl(data_path)
        else:
            # Assume it's a directory with HF dataset
            self._load_hf_dataset(data_path)
    
    def _load_jsonl(self, path: str):

        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", data.get("content", ""))
                if text.strip():
                    self.examples.append(text)
    
    def _load_hf_dataset(self, path: str):
        
        dataset = load_from_disk(path)
        for example in dataset:
            text = example.get("text", example.get("content", ""))
            if text.strip():
                self.examples.append(text)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized data (no tokenization at runtime)."""
    
    def __init__(self, data_path: str):
        """
        Load pre-tokenized data.
        
        Args:
            data_path: Path to directory containing tokenized_data.pt or
                      path to HuggingFace dataset directory with tokenized data
        """
        self.data_path = data_path
        
        # Check if it's a .pt file or directory
        if os.path.isfile(data_path) and data_path.endswith(".pt"):
            # Load from .pt file
            print(f"Loading pre-tokenized data from {data_path}...")
            self.data = torch.load(data_path)
            self.is_hf_dataset = False
        elif os.path.isdir(data_path):
            # Check for tokenized_data.pt in directory
            pt_path = os.path.join(data_path, "tokenized_data.pt")
            if os.path.exists(pt_path):
                print(f"Loading pre-tokenized data from {pt_path}...")
                self.data = torch.load(pt_path)
                self.is_hf_dataset = False
            else:
                # Try loading as HuggingFace dataset
                print(f"Loading pre-tokenized HF dataset from {data_path}...")
                self.data = load_from_disk(data_path)
                # Set format to torch to avoid repeated tensor conversions
                self.data.set_format(type="torch", columns=["input_ids", "attention_mask"])
                self.is_hf_dataset = True
        else:
            raise ValueError(f"Invalid data path: {data_path}")
        
        print(f"Loaded {len(self.data)} pre-tokenized examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.is_hf_dataset:
            example = self.data[idx]
            # Data is already in tensor format due to set_format()
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            
            # Create labels from input_ids if not present
            if "labels" in example:
                labels = example["labels"]
            else:
                labels = input_ids.clone()
                # Mask padding tokens in labels
                labels[attention_mask == 0] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        else:
            # Data is already in tensor format
            return self.data[idx]


def get_lr_scheduler(optimizer, config: TrainConfig, num_training_steps: int):
    """Create learning rate scheduler."""
    
    num_warmup_steps = config.warmup_steps
    
    if config.lr_scheduler == "cosine":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    elif config.lr_scheduler == "linear":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    else:  # constant
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def setup_distributed():
    """Setup distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def is_main_process(rank: int) -> bool:
    return rank == 0


def evaluate(model, eval_loader, device, config: TrainConfig, rank: int):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    # Limit evaluation to eval_batches * gradient_accumulation_steps
    max_eval_batches = config.eval_batches * config.gradient_accumulation_steps
    
    autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    
    with torch.no_grad():
        for batch in eval_loader:
            if num_batches >= max_eval_batches:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]
            
            total_loss += loss.item()
            total_tokens += attention_mask.sum().item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Sync across processes if distributed
    if torch.distributed.is_initialized():
        loss_tensor = torch.tensor([total_loss, num_batches], device=device)
        torch.distributed.all_reduce(loss_tensor)
        avg_loss = loss_tensor[0].item() / loss_tensor[1].item() if loss_tensor[1].item() > 0 else 0.0
    
    # Calculate perplexity
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    model.train()
    return avg_loss, perplexity


def load_checkpoint(checkpoint_path: str, model, optimizer, scheduler, device, rank: int):
    """Load model checkpoint and training state."""
    if is_main_process(rank):
        print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint_dir = Path(checkpoint_path)
    
    # Load model weights
    model_path = checkpoint_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model_state = torch.load(model_path, map_location=device)
    if hasattr(model, "module"):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    
    # Load training state (optimizer, scheduler, step)
    training_state_path = checkpoint_dir / "training_state.pt"
    if not training_state_path.exists():
        raise FileNotFoundError(f"Training state not found: {training_state_path}")
    
    training_state = torch.load(training_state_path, map_location=device)
    optimizer.load_state_dict(training_state["optimizer"])
    scheduler.load_state_dict(training_state["scheduler"])
    step = training_state["step"]
    
    if is_main_process(rank):
        print(f"Resumed from checkpoint at step {step}")
    
    return step


def save_checkpoint(model, optimizer, scheduler, step, config: TrainConfig, rank: int):
    """Save model checkpoint."""
    if not is_main_process(rank):
        return
    
    checkpoint_dir = Path(config.output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model state dict (handle DDP)
    model_to_save = model.module if hasattr(model, "module") else model
    
    # Save model
    torch.save(model_to_save.state_dict(), checkpoint_dir / "model.pt")
    
    # Save optimizer and scheduler
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }, checkpoint_dir / "training_state.pt")
    
    # Save config
    model_to_save.config.to_json = lambda: vars(model_to_save.config)
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(vars(model_to_save.config), f, indent=2, default=str)
    
    print(f"Saved checkpoint to {checkpoint_dir}")


def train(config: TrainConfig):
    """Main training function."""
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    torch.manual_seed(config.seed + rank)
    
    # Generate unique run ID and update output directory
    if config.resume_from_checkpoint:
        # When resuming, use the same output directory as the checkpoint
        config.output_dir = str(Path(config.resume_from_checkpoint).parent)
        if is_main_process(rank):
            print(f"Resuming: Using existing output directory {config.output_dir}")
    else:
        # Create new run with unique ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = os.path.join(config.output_dir, f"run_{run_id}")
        if is_main_process(rank):
            print(f"New run: Checkpoints will be saved to {config.output_dir}")
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if config.wandb_enabled and is_main_process(rank):
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            tags=config.wandb_tags,
            config=vars(config),
        )
    
    # Load tokenizer
    if is_main_process(rank):
        print(f"Loading tokenizer from {config.model_path}...")
    tokenizer = load_tokenizer(config.model_path)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    dtype = torch.bfloat16 if config.model_dtype == "bfloat16" else torch.float16
    
    # Determine model loading strategy
    if config.resume_from_checkpoint:
        # When resuming, load config from checkpoint
        if is_main_process(rank):
            print(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        checkpoint_config_path = os.path.join(config.resume_from_checkpoint, "config.json")
        model_config = Qwen3Config.from_json(checkpoint_config_path)
        model_config.checkpoint_activations = config.checkpoint_activations
        model = Qwen3ForCausalLM(model_config)
        model = model.to(device=device, dtype=dtype)
        # Model weights will be loaded later along with optimizer/scheduler
    elif config.init_from_scratch:
        if is_main_process(rank):
            print(f"Initializing model from scratch (random weights)...")
        # Load config from pretrained path but initialize with random weights
        config_path = os.path.join(config.model_path, "config.json")
        model_config = Qwen3Config.from_json(config_path)
        model_config.checkpoint_activations = config.checkpoint_activations
        model = Qwen3ForCausalLM(model_config)
        model = model.to(device=device, dtype=dtype)
    else:
        if is_main_process(rank):
            print(f"Loading pretrained model from {config.model_path}...")
        model = Qwen3ForCausalLM.from_pretrained(
            config.model_path,
            device=device,
            dtype=dtype,
        )
        model.config.checkpoint_activations = config.checkpoint_activations
        model.model.checkpoint_activations = config.checkpoint_activations
    
    # Print model architecture
    if is_main_process(rank):
        print("\n" + "=" * 80)
        print("MODEL ARCHITECTURE")
        print("=" * 80)
        print(f"Model config:")
        print(f"  vocab_size: {model.config.vocab_size}")
        print(f"  hidden_size: {model.config.hidden_size}")
        print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
        print(f"  num_attention_heads: {model.config.num_attention_heads}")
        print(f"  num_key_value_heads: {model.config.num_key_value_heads}")
        print(f"  head_dim: {model.config.head_dim}")
        print(f"  intermediate_size: {model.config.intermediate_size}")
        print(f"  max_position_embeddings: {model.config.max_position_embeddings}")
        print(f"  rope_theta: {model.config.rope_theta}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nParameter count:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        print(f"  Model size (BF16): {total_params * 2 / 1e9:.2f} GB")
        
        # Print detailed layer summary if torchinfo is available
        if summary is not None and config.verbose:
            print("\n" + "-" * 80)
            print("LAYER-BY-LAYER SUMMARY")
            print("-" * 80)
            try:
                summary(
                    model,
                    input_size=(config.per_device_batch_size, config.max_seq_length),
                    dtypes=[torch.long],
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    depth=3,
                    verbose=config.verbose,
                )
            except Exception as e:
                print(f"Could not generate detailed summary: {e}")
        
        print("=" * 80 + "\n")
    
    # Enable flash attention if requested
    if config.flash_attention:
        _enable_flash_attention(model)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create dataset and dataloader
    if is_main_process(rank):
        print(f"Loading dataset from {config.train_path}...")
    
    # Check if data is pre-tokenized
    is_pretokenized = False
    if os.path.isdir(config.train_path):
        # Check for tokenized_data.pt or HF dataset with tokenized columns
        if os.path.exists(os.path.join(config.train_path, "tokenized_data.pt")):
            is_pretokenized = True
        elif os.path.exists(os.path.join(config.train_path, "dataset_info.json")):
            # HF dataset - check if it has tokenized columns
            try:
                dataset_test = load_from_disk(config.train_path)
                if "input_ids" in dataset_test.column_names:
                    is_pretokenized = True
            except:
                pass
    elif config.train_path.endswith(".pt"):
        is_pretokenized = True
    
    if is_pretokenized:
        if is_main_process(rank):
            print("Using pre-tokenized dataset (no runtime tokenization)")
        full_dataset = PreTokenizedDataset(config.train_path)
    else:
        if is_main_process(rank):
            print("Using raw text dataset (tokenizing at runtime)")
        full_dataset = TextDataset(
            config.train_path,
            tokenizer,
            max_seq_length=config.max_seq_length,
        )
    
    # Do 80/20 train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Set seed for reproducible split
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    if is_main_process(rank):
        print(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Create train dataloader
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.per_device_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )
    
    # Create validation dataloader
    if world_size > 1:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        val_sampler = None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.per_device_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    num_training_steps = config.max_steps or (num_update_steps_per_epoch * config.num_epochs)
    
    if is_main_process(rank):
        print(f"Training examples: {len(train_dataset)}")
        print(f"Validation examples: {len(val_dataset)}")
        print(f"Steps per epoch: {num_update_steps_per_epoch}")
        print(f"Total training steps: {num_training_steps}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Create scheduler
    scheduler = get_lr_scheduler(optimizer, config, num_training_steps)
    
    # Load checkpoint if resuming
    global_step = 0
    if config.resume_from_checkpoint:
        global_step = load_checkpoint(
            config.resume_from_checkpoint,
            model,
            optimizer,
            scheduler,
            device,
            rank
        )
        if is_main_process(rank):
            print(f"Resuming training from step {global_step}")
    
    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if config.mixed_precision == "fp16" else None
    autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    
    # Training loop
    model.train()
    total_loss = 0.0
    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0
    
    if is_main_process(rank):
        print("Starting training...")
    
    for epoch in range(config.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_loader):
            # Move batch to device with non-blocking transfer
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Forward pass with autocast
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"] / config.gradient_accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate UNSCALED loss for logging
            total_loss += outputs["loss"].item()
            
            # Gradient accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if scaler is not None:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                global_step += 1
                
                # Logging
                if global_step % config.logging_steps == 0 and is_main_process(rank):
                    avg_loss = total_loss / (config.logging_steps * config.gradient_accumulation_steps)
                    lr = scheduler.get_last_lr()[0]
                    
                    # Calculate speed metrics
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    
                    # Instantaneous throughput (since last log)
                    time_since_last_log = current_time - last_log_time
                    steps_since_last_log = global_step - last_log_step
                    steps_per_sec = steps_since_last_log / time_since_last_log if time_since_last_log > 0 else 0
                    samples_since_last_log = steps_since_last_log * config.gradient_accumulation_steps * config.per_device_batch_size * world_size
                    samples_per_sec = samples_since_last_log / time_since_last_log if time_since_last_log > 0 else 0
                    ms_per_step = (time_since_last_log / steps_since_last_log * 1000) if steps_since_last_log > 0 else 0
                    
                    # Update last log tracking
                    last_log_time = current_time
                    last_log_step = global_step
                    
                    print(f"Step {global_step}/{num_training_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                          f"{ms_per_step:.1f}ms/step | {samples_per_sec:.1f} samples/s")
                    
                    if config.wandb_enabled:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                            "speed/steps_per_sec": steps_per_sec,
                            "speed/samples_per_sec": samples_per_sec,
                            "speed/ms_per_step": ms_per_step,
                        }, step=global_step)
                    
                    total_loss = 0.0
                
                # Evaluation
                if global_step % config.eval_steps == 0 and is_main_process(rank):
                    eval_loss, eval_perplexity = evaluate(model, val_loader, device, config, rank)
                    print(f"Validation at step {global_step}: Loss = {eval_loss:.4f}, Perplexity = {eval_perplexity:.2f}")
                    
                    if config.wandb_enabled:
                        wandb.log({
                            "eval/loss": eval_loss,
                            "eval/perplexity": eval_perplexity,
                        }, step=global_step)
                
                # Save checkpoint
                if global_step % config.save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, config, rank)
                
                # Check max steps
                if config.max_steps and global_step >= config.max_steps:
                    break
        
        if config.max_steps and global_step >= config.max_steps:
            break
    
    # Final save
    save_checkpoint(model, optimizer, scheduler, global_step, config, rank)
    
    if config.wandb_enabled and is_main_process(rank):
        
        wandb.finish()
    
    if world_size > 1:
        dist.destroy_process_group()
    
    if is_main_process(rank):
        print("Training complete!")


def _enable_flash_attention(model):
    """Enable flash attention for the model."""

    # Monkey-patch the attention forward
    for layer in model.model.layers:
        layer.self_attn._use_flash_attention = True
        layer.self_attn._flash_attn_func = flash_attn_func
    print("Flash attention enabled")



def main():
    parser = argparse.ArgumentParser(description="Train Qwen3 model")
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.json",
        help="Path to training config JSON",
    )
    parser.add_argument(
        "--download_data",
        action="store_true",
        help="Download and tokenize dataset before training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pile",
        help="Dataset to download (pile, wikitext, etc.)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of shards to download (for Pile dataset)",
    )
    parser.add_argument(
        "--tokenize_workers",
        type=int,
        default=None,
        help="Number of parallel workers for tokenization",
    )
    args = parser.parse_args()
    
    config = TrainConfig.from_json(args.config)
    
    # Download and tokenize data if requested
    if args.download_data:
        raise NotImplementedError("Please download the data manually by running the download_and_tokenize.py script, as this functionality has been moved to a separate script.")
    
    train(config)


if __name__ == "__main__":
    main()
