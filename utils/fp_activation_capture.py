"""Capture one sequence from an unwrapped FP decoder, without PTQ or PPL."""

import json
from pathlib import Path
import random
import time

import torch


def select_sequence(input_ids, sequence_length, seed, sequence_index=None):
    """Select a complete, nonoverlapping window from an unpadded token stream."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Expected a single unpadded token stream [1, tokens]")
    if sequence_length < 1:
        raise ValueError("Sequence length must be positive")
    count = input_ids.shape[1] // sequence_length
    if count == 0:
        raise ValueError("Input does not contain a complete sequence")
    index = random.Random(seed).randrange(count) if sequence_index is None else sequence_index
    if not 0 <= index < count:
        raise ValueError(f"Sequence index must be in [0, {count})")
    start = index * sequence_length
    return input_ids[:, start:start + sequence_length].detach().cpu().clone(), {
        "source_sequence_index": index,
        "source_num_sequences": count,
        "source_token_offset": start,
        "source_num_tokens": input_ids.shape[1],
        "source_tail_tokens": input_ids.shape[1] % sequence_length,
        "seed": seed,
        "sequence_selection": "seeded random window" if sequence_index is None else "explicit window",
    }


def validate_fp_model(model, allow_residual_rotations=False):
    """Reject known quantized/rotated modules; fused checkpoint history is external."""
    if getattr(model, "is_quantized", False) or getattr(model.config, "quantization_config", None):
        raise ValueError("FP capture requires an unquantized checkpoint")
    if getattr(model.config, "pretraining_tp", 1) != 1:
        raise ValueError("FP capture requires pretraining_tp=1 so projection hooks execute")
    for module in model.modules():
        if any(hasattr(module, name) for name in ("quantizer", "out_quantizer", "k_quantizer")):
            raise ValueError("FP capture requires a model without quantization wrappers")
        if any(getattr(module, name, False) for name in ("online_full_had", "online_partial_had")):
            raise ValueError("Online rotations must be disabled for FP capture")
        forbidden = ("R1", "R2") if allow_residual_rotations else (
            "attn_residual_rotation", "mlp_residual_rotation", "R1", "R2")
        if any(getattr(module, name, None) is not None for name in forbidden):
            raise ValueError("Rotation modules/buffers must be absent for FP capture")
    for layer in model.model.layers:
        for parent, names in (
            (layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
            (layer.mlp, ("gate_proj", "up_proj", "down_proj")),
        ):
            if any(type(getattr(parent, name)) is not torch.nn.Linear for name in names):
                raise ValueError("FP capture requires plain torch.nn.Linear projections")


@torch.no_grad()
def capture_fp_activations(model, input_ids, output_dir, metadata=None):
    """Run only the backbone once; save actual Q/gate Linear inputs after RMSNorm.

    input_ids must be one complete, unpadded sequence. No fusion, rotation,
    quantization, calibration, or language-model head evaluation is performed.
    """
    validate_fp_model(model)
    return _capture_projection_inputs(model, input_ids, output_dir, metadata)


@torch.no_grad()
def _capture_projection_inputs(model, input_ids, output_dir, metadata=None, profile=None):
    """Shared writer. Callers must validate their preparation and model first."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("FP capture expects one nonempty sequence [1, tokens]")
    layers = model.model.layers
    if not len(layers):
        raise ValueError("Model has no decoder layers")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    token_ids = input_ids.detach().cpu().clone()
    torch.save({"input_ids": token_ids}, directory / "tokens.pt")
    manifest = {
        **(metadata or {}),
        "schema_version": 1,
        "capture_mode": "fp_single_sequence",
        "status": "collecting",
        "num_layers": len(layers),
        "num_sequences": 1,
        "sequence_length": token_ids.shape[1],
        "num_tokens": token_ids.numel(),
        "activation_sequence_index": 0,
        "sequence_order": "x token order matches tokens.pt input_ids[0]",
        "rotate": False, "norm_fused": False, "qk_hadamard": False,
        "w_bits": 16, "a_bits": 16, "k_bits": 16, "v_bits": 16,
        "activation_basis": "original channels, post-RMSNorm with gamma retained",
        "activation_dtype": str(model.model.embed_tokens.weight.dtype),
        "model_device": str(model.model.embed_tokens.weight.device),
        "statistics_dtype": "torch.float32",
        "statistics_scope": "all tokens of the single saved sequence",
        "padding": False, "use_cache": False,
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "sites": {"attn": "self_attn.q_proj input", "ffn": "mlp.gate_proj input"},
        "calibration_performed": False, "ppl_evaluated": False,
    }
    manifest.update(profile or {})

    def save_manifest():
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    save_manifest()
    seen = set()
    handles = []

    def make_hook(layer_index, site):
        def hook(module, inputs):
            started = time.perf_counter()
            key = (layer_index, site)
            if key in seen:
                raise RuntimeError(f"Duplicate activation capture: {key}")
            x = inputs[0].detach()
            expected = (1, token_ids.shape[1], module.in_features)
            if tuple(x.shape) != expected:
                raise RuntimeError(f"{key}: expected {expected}, got {tuple(x.shape)}")
            print(f"[FP capture] Layer {layer_index} {site}: input on {x.device}; "
                  "copying to CPU, computing statistics and saving", flush=True)
            raw = x[0].to(device="cpu").clone()
            values = raw.float()
            if not torch.isfinite(values).all():
                raise ValueError(f"Nonfinite activation at layer {layer_index}, site {site}")
            stats = {
                "channel_max_abs": values.abs().amax(dim=0),
                "channel_rms": values.square().mean(dim=0).sqrt(),
                "token_max_abs": values.abs().amax(dim=-1),
                "x_norm": torch.linalg.vector_norm(values, dim=-1).unsqueeze(0),
            }
            if not all(torch.isfinite(value).all() for value in stats.values()):
                raise ValueError(f"Nonfinite statistic at layer {layer_index}, site {site}")
            torch.save({
                "layer_index": layer_index, "site": site,
                "capture_mode": manifest["capture_mode"], "num_sequences": 1,
                "num_tokens": raw.shape[0], "activation_sequence_index": 0,
                "activation_dtype": str(raw.dtype), "x": raw, **stats,
            }, directory / f"layer_{layer_index:03d}_{site}.pt")
            seen.add(key)
            print(f"[FP capture] Layer {layer_index} {site}: saved "
                  f"({time.perf_counter() - started:.2f}s)", flush=True)
            # Returning None preserves the Linear input.
        return hook

    was_training = model.training
    model.eval()
    try:
        for index, layer in enumerate(layers):
            for site, projection in (("attn", layer.self_attn.q_proj), ("ffn", layer.mlp.gate_proj)):
                handles.append(projection.register_forward_pre_hook(make_hook(index, site)))
        device = model.model.embed_tokens.weight.device
        started = time.perf_counter()
        print(f"[FP capture] START single-sequence backbone forward on {device}", flush=True)
        model.model(input_ids=token_ids.to(device), use_cache=False)
        print(f"[FP capture] DONE backbone forward and saves "
              f"({time.perf_counter() - started:.2f}s)", flush=True)
        expected_sites = {(index, site) for index in range(len(layers)) for site in ("attn", "ffn")}
        if seen != expected_sites:
            raise RuntimeError(f"Missing activation hooks: {sorted(expected_sites - seen)}")
        manifest["status"] = "complete"
        manifest["hook_calls"] = len(seen)
        save_manifest()
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        save_manifest()
        raise
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return manifest
