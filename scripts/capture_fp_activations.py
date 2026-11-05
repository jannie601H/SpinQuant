#!/usr/bin/env python3
"""Extract one WikiText2 sequence's FP activations without rotation, PTQ or PPL."""

import argparse
from contextlib import contextmanager
import faulthandler
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@contextmanager
def stage(name):
    started = time.perf_counter()
    print(f"[FP capture] START {name}", flush=True)
    yield
    print(f"[FP capture] DONE  {name} ({time.perf_counter() - started:.2f}s)", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--sequence-index", type=int, default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.sequence_length < 1:
        parser.error("--sequence-length must be positive")
    if args.sequence_index is not None and args.sequence_index < 0:
        parser.error("--sequence-index must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output_dir}")
    print("[FP capture] Importing dependencies", flush=True)
    import torch
    import transformers
    from transformers import AutoConfig, LlamaTokenizerFast
    from eval_utils.modeling_llama import LlamaForCausalLM
    from utils import data_utils
    from utils.fp_activation_capture import capture_fp_activations, select_sequence

    transformers.set_seed(args.seed)
    print(f"[FP capture] Requested device={args.device}, dtype={args.dtype}; "
          f"CUDA available={torch.cuda.is_available()}", flush=True)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    with stage("load model config"):
        config = AutoConfig.from_pretrained(args.input_model)
    if getattr(config, "quantization_config", None):
        raise ValueError("Use an original, unquantized model checkpoint")
    with stage("load tokenizer"):
        tokenizer = LlamaTokenizerFast.from_pretrained(
            args.input_model, model_max_length=args.sequence_length,
            padding_side="right", add_eos_token=False, add_bos_token=False,
        )
    with stage("load WikiText2 and tokenize test stream (CPU/network)"):
        encoding = data_utils.get_wikitext2(tokenizer=tokenizer, eval_mode=True)
    input_ids, selection = select_sequence(
        encoding.input_ids, args.sequence_length, args.seed, args.sequence_index,
    )
    del encoding
    print(f"[FP capture] Selected sequence={selection['source_sequence_index']}, "
          f"tokens={input_ids.shape[1]}", flush=True)
    # Load fresh weights and call the decoder backbone directly. Never call
    # ptq_model/prepare_model or fuse_layer_norms in this extraction path.
    with stage("load model weights (CPU/network)"):
        model = LlamaForCausalLM.from_pretrained(
            args.input_model, config=config, torch_dtype=getattr(torch, args.dtype),
        )
    with stage(f"move model to {args.device}"):
        model = model.to(args.device).eval()
    print(f"[FP capture] Actual model device={model.model.embed_tokens.weight.device}", flush=True)
    manifest = capture_fp_activations(model, input_ids, args.output_dir, metadata={
        **selection, "model": args.input_model, "tokenizer": tokenizer.name_or_path,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "dataset": "Salesforce/wikitext", "subset": "wikitext-2-raw-v1", "split": "test",
        "add_bos_token": tokenizer.add_bos_token, "add_eos_token": tokenizer.add_eos_token,
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
    })
    print(f"Saved FP activations: {args.output_dir}")
    print(f"Source sequence {manifest['source_sequence_index']}; "
          f"{manifest['num_layers']} layers; {manifest['num_tokens']} tokens; "
          f"{manifest['activation_dtype']}; no quantization or PPL")
    return 0


if __name__ == "__main__":
    # Emit diagnostic stacks even during blocking downloads/file locks. This
    # neither cancels the run nor changes model computations.
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    try:
        raise SystemExit(main())
    finally:
        faulthandler.cancel_dump_traceback_later()
