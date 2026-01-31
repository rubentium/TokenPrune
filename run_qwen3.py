"""
Example script showing how to load and run the Qwen3-0.6B-Base model.
"""

import sys
import os

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import Qwen3Config
from model import Qwen3ForCausalLM, load_tokenizer


def main():
    # Path to the model checkpoint
    model_path = "../qwen3_06b_base"
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model...")
    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)
    
    # Example 1: Greedy decoding
    prompt = "The quick brown fox"
    print(f"\n{'='*50}")
    print(f"Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    print("\nGreedy decoding:")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,
        )
    print(f"  {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")
    
    # Example 2: Sampling with temperature
    print("\nSampling (temperature=0.7, top_p=0.9):")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
        )
    print(f"  {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")
    
    # Example 3: Get logits/hidden states for custom processing
    print(f"\n{'='*50}")
    print("Forward pass (for custom processing):")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
        )
    
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Number of hidden state layers: {len(outputs['hidden_states'])}")
    print(f"  Number of attention layers: {len(outputs['attentions'])}")
    print(f"  Predicted next token: '{tokenizer.decode(outputs['logits'][0, -1].argmax().item())}'")


if __name__ == "__main__":
    main()
