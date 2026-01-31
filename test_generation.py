"""
Test script for Qwen3 generation.

Usage:
    python3 test_generation.py
    python3 test_generation.py --prompt "Your custom prompt"
    python3 test_generation.py --max_tokens 100 --temperature 0.8
"""

import argparse
import sys
import os

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import Qwen3Config
from model import Qwen3ForCausalLM, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3 generation")
    parser.add_argument("--model_path", type=str, default="../qwen3_06b_base",
                        help="Path to the model checkpoint")
    parser.add_argument("--prompt", type=str, default="The quick brown fox",
                        help="Input prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) filtering")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of sampling")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu, auto-detected if not specified)")
    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_path)

    # Tokenize input
    print(f"\nPrompt: {args.prompt}")
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"Input tokens: {input_ids.shape[1]}")

    # Generate
    print("\nGenerating...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature if not args.greedy else 1.0,
            top_k=args.top_k if not args.greedy else 0,
            top_p=args.top_p if not args.greedy else 1.0,
            do_sample=not args.greedy,
        )

    # Decode and print
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    
    print(f"\nGenerated ({new_tokens} new tokens):")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)


if __name__ == "__main__":
    main()
