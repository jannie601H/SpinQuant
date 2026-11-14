#!/usr/bin/env python3
"""Capture post-fusion FP activations: 64 grouping train + 32 grouping test blocks."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--train-sequences", type=int, default=64)
    parser.add_argument("--test-sequences", type=int, default=32)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if min(args.sequence_length, args.train_sequences, args.test_sequences) < 1:
        parser.error("sequence length and train/test counts must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output_dir}")
    import datasets
    import torch
    import transformers
    from transformers import AutoConfig, LlamaTokenizerFast
    from eval_utils.modeling_llama import LlamaForCausalLM
    from utils.data_utils import CustomJsonDataset
    from utils.multiseq_activation_capture import select_blocks, capture_fused_sequences

    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    transformers.set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.input_model)
    if getattr(config, "quantization_config", None):
        raise ValueError("Use an original, unquantized model checkpoint")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        args.input_model, model_max_length=args.sequence_length, padding_side="right",
        add_eos_token=False, add_bos_token=False,
    )
    print("[Fused FP capture] Load WikiText-2 TRAIN and build training-compatible blocks", flush=True)
    dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    fingerprint = getattr(dataset, "_fingerprint", None)
    blocks = CustomJsonDataset(dataset, tokenizer, block_size=args.sequence_length)
    tokens, selection = select_blocks(blocks.input_ids, args.sequence_length, args.seed,
                                      args.train_sequences, args.test_sequences)
    del blocks, dataset
    print(f"[Fused FP capture] Selected {len(tokens)} blocks; load fresh model", flush=True)
    model = LlamaForCausalLM.from_pretrained(args.input_model, config=config,
                                            torch_dtype=getattr(torch, args.dtype))
    model = model.to(args.device).eval()
    manifest = capture_fused_sequences(model, tokens, selection, args.output_dir, metadata={
        "model": args.input_model, "model_revision": getattr(config, "_commit_hash", None),
        "tokenizer": tokenizer.name_or_path, "dataset_fingerprint": fingerprint,
        "add_bos_token": tokenizer.add_bos_token, "add_eos_token": tokenizer.add_eos_token,
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
    })
    print(f"Saved {manifest['completed_sequences']} post-fusion sequences to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
