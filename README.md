# TokenPrune - Qwen3 Model Implementation

A modular implementation of the Qwen3 architecture designed for easy architectural experimentation and token pruning research.

## Quick Start

### 1. Setup Data (Download + Tokenize)

#### Option A: Using the dedicated download script (Recommended)

```bash
# Download and tokenize The Pile dataset (5 shards for testing)
python download_and_tokenize.py \
    --dataset pile \
    --output ./data/pile_train \
    --num_shards 5 \
    --num_workers 8

# Or use the convenience script
./download_pile.sh
```

#### Option B: Download and train in one command

```bash
# Download, tokenize, and train in one command
python train.py \
    --config train_config.json \
    --download_data \
    --dataset pile \
    --num_shards 5 \
    --tokenize_workers 8
```

#### Option C: Quick sample data for testing

```bash
# Download WikiText dataset (smaller, faster)
python download_and_tokenize.py \
    --dataset wikitext \
    --output ./data/wikitext \
    --num_workers 4
```

**See [DATASET_GUIDE.md](DATASET_GUIDE.md) for detailed documentation.**

### 2. Train the Model

```bash
# Single GPU
python train.py --config train_config.json

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 train.py --config train_config.json
```

### 3. Generate Text

```bash
python run_qwen3.py
```

Or use in your code:

```python
import torch
from TokenPrune import Qwen3ForCausalLM, load_tokenizer

# Load model
model_path = "../qwen3_06b_base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Qwen3ForCausalLM.from_pretrained(model_path, device=device)
tokenizer = load_tokenizer(model_path)

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

output_ids = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Training Features

- ✅ **Parallel data preprocessing** - Multi-process tokenization for fast data prep
- ✅ **Flexible dataset support** - The Pile, WikiText, OpenWebText, or any HuggingFace dataset
- ✅ **Shard-based downloading** - Download only what you need (5-10 shards recommended)
- ✅ **Pre-tokenized data** - 5-10x faster data loading during training
- ✅ **Multi-GPU training** - DDP support via torchrun
- ✅ **Flash Attention** - Memory efficient attention
- ✅ **WandB logging** - Track your experiments
- ✅ **Mixed precision** - BF16/FP16 training
- ✅ **Gradient accumulation** - Train with larger effective batch sizes
- ✅ **Integrated pipeline** - Download, tokenize, and train in one command

## Architecture Overview

The implementation is modular and follows the Qwen3 architecture:

- **Qwen3Config**: Configuration dataclass with all model hyperparameters
- **Qwen3RMSNorm**: Root Mean Square Layer Normalization
- **Qwen3RotaryEmbedding**: Rotary Position Embeddings (RoPE)
- **Qwen3MLP**: SwiGLU-based feedforward network
- **Qwen3Attention**: Multi-head attention with Grouped Query Attention (GQA)
- **Qwen3DecoderLayer**: Single transformer decoder layer
- **Qwen3Model**: The base transformer model
- **Qwen3ForCausalLM**: Model with language modeling head

## Key Features

- **Grouped Query Attention (GQA)**: 16 query heads, 8 KV heads
- **RoPE**: Rotary position embeddings with θ=1,000,000
- **SwiGLU Activation**: In the MLP layers
- **QK Normalization**: RMSNorm applied to Q and K before attention
- **Tied Embeddings**: Input and output embeddings are shared

## Making Architectural Modifications

### Example: Modifying Attention

```python
from TokenPrune import Qwen3Attention, Qwen3Config

class CustomAttention(Qwen3Attention):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, 
                past_key_value=None, use_cache=False, output_attentions=False):
        # Your custom attention logic here
        # For example, add token pruning
        
        # Call parent implementation or implement your own
        return super().forward(
            hidden_states, attention_mask, position_ids,
            past_key_value, use_cache, output_attentions
        )
```

### Example: Modifying the Decoder Layer

```python
from TokenPrune import Qwen3DecoderLayer

class CustomDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Add custom components
        self.token_selector = YourTokenSelector(config)
    
    def forward(self, hidden_states, **kwargs):
        # Add token selection/pruning logic
        hidden_states, mask = self.token_selector(hidden_states)
        return super().forward(hidden_states, **kwargs)
```

## Model Configuration (Qwen3-0.6B-Base)

| Parameter | Value |
|-----------|-------|
| Hidden Size | 1024 |
| Intermediate Size | 3072 |
| Num Layers | 28 |
| Num Attention Heads | 16 |
| Num KV Heads | 8 |
| Head Dim | 128 |
| Vocab Size | 151936 |
| Max Position Embeddings | 32768 |

## Dependencies

```bash
pip install torch transformers safetensors datasets wandb tqdm
```

## File Structure

```
TokenPrune/
├── __init__.py                 # Package exports
├── config.py                   # Model configuration
├── attention.py                # Attention and normalization layers
├── model.py                    # Core model implementation
├── download_and_tokenize.py    # NEW: Parallel dataset download + tokenization
├── download_pile.sh            # NEW: Convenience script for The Pile
├── test_download.py            # NEW: Test download functionality
├── train.py                    # Training script with multi-GPU support
├── train_config.json           # Training configuration
├── run_qwen3.py                # Example usage script
├── test_generation.py          # Test generation
├── DATASET_GUIDE.md            # NEW: Detailed dataset documentation
└── README.md                   # This file
```

## Data Download and Tokenization

The new `download_and_tokenize.py` script provides:

- **Parallel tokenization** using multiprocessing
- **Shard-based downloads** to avoid downloading 800GB+ at once
- **Multiple datasets** support (Pile, WikiText, OpenWebText, custom)
- **Flexible formats** (HuggingFace or PyTorch)
- **Integration with training** via command-line flags

### Quick Examples

```bash
# Download 5 shards of The Pile with 8 parallel workers
python download_and_tokenize.py --dataset pile --output ./data/pile --num_shards 5 --num_workers 8

# Download WikiText dataset
python download_and_tokenize.py --dataset wikitext --output ./data/wikitext

# Use custom tokenizer and sequence length
python download_and_tokenize.py \
    --dataset pile \
    --output ./data \
    --tokenizer ./my_tokenizer \
    --max_length 4096 \
    --num_shards 10
```

### Integration with Training

```bash
# Download and train in one command
python train.py --config train_config.json --download_data --dataset pile --num_shards 5

# Multi-GPU training with auto-download
torchrun --nproc_per_node=4 train.py --config train_config.json --download_data --dataset pile --num_shards 10
```

**For complete documentation, see [DATASET_GUIDE.md](DATASET_GUIDE.md)**

## Dataset Options

### The Pile
- **Size**: 800GB+ (full dataset)
- **Recommended**: Use 5-10 shards for development (~50-100k examples)
- **Usage**: `--dataset pile --num_shards 5`

### WikiText
- **Size**: ~500MB
- **Good for**: Quick experiments and testing
- **Usage**: `--dataset wikitext`

### OpenWebText
- **Size**: ~12GB  
- **Good for**: Medium-scale pre-training
- **Usage**: `--dataset openwebtext`

### Custom Datasets
Any HuggingFace dataset:
```bash
python download_and_tokenize.py --dataset your-username/your-dataset --output ./data
```
