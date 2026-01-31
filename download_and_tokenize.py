import argparse
import os
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def get_stream_generator(dataset_name, split, num_samples):
    """Yields exactly num_samples from the streaming dataset."""
    ds = load_dataset(dataset_name, split=split, streaming=True)
    # We yield only what we need to avoid massive downloads
    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        yield example

def main():
    parser = argparse.ArgumentParser(description="Efficient Dataset Tokenizer")
    parser.add_argument("--dataset", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="../qwen3_06b_base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_shards", type=int, default=5, help="Units of 10k rows")
    parser.add_argument("--num_proc", type=int, default=os.cpu_count() - 1)
    args = parser.parse_args()

    # 1. Setup paths and Tokenizer
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Materialize the subset (Download phase)
    # This only pulls the rows we specify, preventing the 800GB issue
    total_samples = args.num_shards * 10000
    print(f"--- Downloading {total_samples} samples from {args.dataset} ---")
    
    # from_generator converts the stream into a local 'Map-style' dataset 
    # that supports native multiprocessing in the next step
    raw_ds = Dataset.from_generator(
        get_stream_generator, 
        gen_kwargs={"dataset_name": args.dataset, "split": "train", "num_samples": total_samples}
    )

    # 3. Parallel Tokenization (Processing phase)
    print(f"--- Tokenizing with {args.num_proc} workers ---")
    
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length"
        )

    tokenized_ds = raw_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=raw_ds.column_names, # Drop text to save disk space
        desc="Tokenizing"
    )

    # 4. Save (Storage phase)
    print(f"--- Saving to {out_dir} ---")
    tokenized_ds.save_to_disk(str(out_dir / "tokenized"))
    
    # Save a small metadata file for your records
    import json
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "samples": len(tokenized_ds),
            "max_length": args.max_length,
            "tokenizer": args.tokenizer
        }, f)

    print(f"Done! Total examples: {len(tokenized_ds)}")

if __name__ == "__main__":
    main()