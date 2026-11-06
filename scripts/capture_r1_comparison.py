#!/usr/bin/env python3
"""Capture original, gamma-fused, global R1 and layer-wise R1 FP inputs on identical tokens."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=Path, required=True, help="Existing complete FP capture with tokens.pt")
    p.add_argument("--global-rotation", type=Path, required=True)
    p.add_argument("--layerwise-rotation", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="Must not exist")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--fusion-dtype", choices=("float32", "float64"), default="float32")
    args = p.parse_args(argv)
    if args.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    return args


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output_dir}")
    import torch
    import transformers
    from transformers import AutoConfig
    from eval_utils.modeling_llama import LlamaForCausalLM
    from utils.fp_activation_capture import capture_fp_activations
    from utils.r1_activation_capture import (
        checkpoint_rotations, fuse_norm_scales_only, apply_r1_only,
        capture_r1_inputs, distribution_metrics, difference_metrics,
    )

    torch.set_num_threads(args.cpu_threads)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = json.loads((args.reference_dir / "manifest.json").read_text())
    if (ref.get("status") != "complete" or ref.get("capture_mode") != "fp_single_sequence"
            or ref.get("rotate") or ref.get("norm_fused") or ref.get("padding")
            or any(ref.get(k) != 16 for k in ("w_bits", "a_bits", "k_bits", "v_bits"))):
        raise ValueError("Reference must be a complete original FP, unpadded, unrotated capture")
    ids = torch.load(args.reference_dir / "tokens.pt", map_location="cpu", weights_only=True)["input_ids"]
    if ids.dtype != torch.int64 or tuple(ids.shape) != (1, ref["sequence_length"]):
        raise ValueError("Invalid reference token shape or dtype")
    dtype_name = ref["activation_dtype"].replace("torch.", "")
    if dtype_name not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported reference dtype")
    transformers.set_seed(ref["seed"])
    pretrained = {"revision": ref["model_revision"]} if ref.get("model_revision") else {}
    print("[R1 capture] Loading pinned FP model; reusing tokens.pt, no tokenizer/dataset loading", flush=True)
    config = AutoConfig.from_pretrained(ref["model"], **pretrained)
    if getattr(config, "quantization_config", None) or getattr(config, "pretraining_tp", 1) != 1:
        raise ValueError("Use an unquantized model with pretraining_tp=1")
    if config.num_hidden_layers != ref["num_layers"]:
        raise ValueError("Model/reference layer counts differ")
    checkpoints, provenance = {}, {}
    for mode, path in (("global", args.global_rotation), ("layerwise", args.layerwise_rotation)):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoints[mode] = checkpoint_rotations(checkpoint, mode, config.num_hidden_layers, config.hidden_size)
        result_path = path.parent / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if result.get("conditions", {}).get("model", ref["model"]) != ref["model"]:
            raise ValueError(f"{mode} checkpoint model differs from reference")
        provenance[mode] = {"path": str(path.resolve()), "sha256": sha256(path),
                            "training_conditions": result.get("conditions"),
                            "training_command": result.get("commands", {}).get("train")}
        del checkpoint
    model = LlamaForCausalLM.from_pretrained(
        ref["model"], config=config, torch_dtype=getattr(torch, dtype_name),
        attn_implementation=ref.get("attention_implementation", "sdpa"), **pretrained,
    ).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "collecting", "schema_version": 1, "reference": str(args.reference_dir.resolve()),
        "reference_manifest": ref, "rotation_checkpoints": provenance,
        "tokens_file_sha256": sha256(args.reference_dir / "tokens.pt"),
        "modes": ["original", "norm_fused", "global", "layerwise"],
        "scope": "R1 only; actual FP model forwards at Q/gate inputs",
        "quantization": False, "embedding_centered": False, "r2_r3_r4_applied": False,
        "forward_dtype": dtype_name, "fusion_dtype": args.fusion_dtype,
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
        "device": args.device, "checks": {},
    }

    def save_report():
        (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save_report()
    common = {k: ref[k] for k in ("model", "model_revision", "seed", "source_sequence_index",
                                  "source_token_offset", "dataset", "subset", "split",
                                  "add_bos_token", "add_eos_token") if k in ref}
    common.update(torch_version=str(torch.__version__), transformers_version=transformers.__version__,
                  reference_tokens_sha256=report["tokens_file_sha256"])
    original_logits = fused_hidden = baseline = None
    try:
        for mode in report["modes"]:
            print(f"[R1 capture] START {mode}", flush=True)
            if mode == "norm_fused":
                model.cpu()
                fuse_norm_scales_only(model)
                baseline = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif mode in checkpoints:
                model.cpu()
                for layer in model.model.layers:
                    layer.attn_residual_rotation = layer.mlp_residual_rotation = None
                model.load_state_dict(baseline, strict=True)
                report["checks"][mode + "_orthogonality"] = apply_r1_only(
                    model, checkpoints[mode], args.device, getattr(torch, args.fusion_dtype))
            model.to(args.device).eval()
            outputs = []

            def record_output(module, inputs, output):
                h = output if torch.is_tensor(output) else (
                    output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0])
                outputs.append(h.detach().cpu())

            handle = model.model.register_forward_hook(record_output)
            try:
                metadata = {**common, "rotation_checkpoint": provenance.get(mode)}
                if mode == "original":
                    capture_fp_activations(model, ids, args.output_dir / mode, metadata)
                else:
                    capture_r1_inputs(model, ids, args.output_dir / mode, mode, metadata)
            finally:
                handle.remove()
            if len(outputs) != 1:
                raise RuntimeError("Expected exactly one backbone forward")
            hidden = outputs.pop()
            with torch.no_grad():
                # One token only: verify output preservation without full logits or PPL.
                logits = model.lm_head(hidden[:, -1:].to(args.device)).float().cpu()
            if mode == "original":
                original_logits = logits
            else:
                report["checks"][mode + "_last_token_logits_vs_original"] = difference_metrics(logits, original_logits)
            if mode == "norm_fused":
                fused_hidden = hidden.float()
            elif mode in checkpoints:
                canonical = hidden.float().to(args.device) @ checkpoints[mode][-1].float().to(args.device).T
                report["checks"][mode + "_backbone_unrotated_vs_norm_fused"] = difference_metrics(canonical.cpu(), fused_hidden)
            save_report()
        rows = []
        for i in range(config.num_hidden_layers):
            for site_index, site in enumerate(("attn", "ffn")):
                name = f"layer_{i:03d}_{site}.pt"
                base = torch.load(args.output_dir / "norm_fused" / name, weights_only=True)["x"].float()
                for mode in report["modes"]:
                    x = torch.load(args.output_dir / mode / name, weights_only=True)["x"]
                    row = {"layer": i, "site": site, "mode": mode, **distribution_metrics(x)}
                    if mode in checkpoints:
                        with torch.no_grad():
                            expected = base.to(args.device) @ checkpoints[mode][2 * i + site_index].float().to(args.device)
                        row.update(difference_metrics(x, expected.cpu()))
                    elif mode == "original":
                        previous = torch.load(args.reference_dir / name, map_location="cpu", weights_only=True)["x"]
                        row.update(difference_metrics(x, previous))
                    rows.append(row)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (args.output_dir / "distribution.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        report["metric_comparison"] = "Use norm_fused vs global/layerwise for rotation redistribution; original retains gamma."
        report["difference_metrics"] = "Original: replay vs reference. Rotated sites: actual input vs norm_fused input @ R in FP32. Relative error is Frobenius/L2."
        report["checks_note"] = "FP rounding discrepancies are reported, not automatically certified as negligible; inspect before interpreting redistribution."
        report["status"] = "complete"
        save_report()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save_report()
        raise
    print(f"[R1 capture] Complete: {args.output_dir}; inspect comparison.json and distribution.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
