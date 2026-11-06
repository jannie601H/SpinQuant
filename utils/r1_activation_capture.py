"""FP R1 interventions for Q/gate input analysis; no PTQ or online Hadamard."""

import torch

from utils.fp_activation_capture import validate_fp_model, _capture_projection_inputs


def checkpoint_rotations(checkpoint, layout, num_layers, hidden_size):
    """Resolve legacy R1/A/B or equivalent grouped layouts, rejecting wrong layouts."""
    if layout not in ("global", "layerwise"):
        raise ValueError("layout must be global or layerwise")
    meta = checkpoint.get("_rotation_metadata")
    if meta is not None:
        ids = meta.get("group_indices")
        expected = ([0] * (2 * num_layers + 1) if layout == "global" else
                    list(range(2 * num_layers)) + [2 * num_layers - 1])
        if meta.get("num_hidden_layers") != num_layers or ids != expected:
            raise ValueError(f"Checkpoint is not a {layout} layout for {num_layers} layers")
        rotations = [checkpoint[f"Q.{g}"] for g in ids]
    elif layout == "global":
        if "R1" not in checkpoint or any(k.startswith(("A.", "B.", "Q.")) for k in checkpoint):
            raise ValueError("Expected a global R1 checkpoint")
        rotations = [checkpoint["R1"]] * (2 * num_layers + 1)
    else:
        if "R1" in checkpoint or any(k.startswith("Q.") for k in checkpoint):
            raise ValueError("Expected a layer-wise A/B checkpoint")
        keys = [k for i in range(num_layers) for k in (f"A.{i}", f"B.{i}")] + [f"A.{num_layers}"]
        rotations = [checkpoint[k] for k in keys]
    for r in rotations:
        if not torch.is_tensor(r) or r.shape != (hidden_size, hidden_size):
            raise ValueError("Rotation dimension does not match model hidden_size")
        if not r.is_floating_point() or not torch.isfinite(r).all():
            raise ValueError("Rotation must be a finite floating-point matrix")
    return rotations


@torch.no_grad()
def fuse_norm_scales_only(model):
    """Fold gamma into consumers, preserving embeddings and the FP function.

    Deliberately excludes the embedding centering in fuse_layer_norms. Final
    gamma is folded into an untied head so shared embeddings cannot be changed.
    """
    validate_fp_model(model)
    from utils.fuse_norm_utils import fuse_ln_linear
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings = False
    pairs = []
    for layer in model.model.layers:
        pairs.extend([
            (layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]),
            (layer.post_attention_layernorm, [layer.mlp.gate_proj, layer.mlp.up_proj]),
        ])
    pairs.append((model.model.norm, [model.lm_head]))
    for norm, projections in pairs:
        fuse_ln_linear(norm, projections)
        norm.weight.fill_(1)


@torch.no_grad()
def apply_r1_only(model, rotations, device, compute_dtype=torch.float32, orthogonality_tol=1e-3):
    """Apply the residual R1 family to a gamma-fused FP model in memory.

    Input weights W -> W R; output weights W -> R_out.T W; residual
    transitions R_in.T R_out. R2/R3/R4 and embedding centering are excluded.
    No stored model weights or rotation checkpoint are changed.
    """
    validate_fp_model(model)
    layers = model.model.layers
    if len(rotations) != 2 * len(layers) + 1:
        raise ValueError("Expected one rotation per A/F input plus final output")
    norms = [model.model.norm] + [n for l in layers for n in (l.input_layernorm, l.post_attention_layernorm)]
    if any(not torch.equal(n.weight, torch.ones_like(n.weight)) for n in norms):
        raise ValueError("Fuse norm scales before applying R1")
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        raise ValueError("Untie the head before rotating weights")
    diagnostics = []
    checked = set()
    for r in rotations:
        if id(r) in checked:
            continue
        checked.add(id(r))
        q = r.to(device=device, dtype=compute_dtype)
        if q.shape != (model.config.hidden_size, model.config.hidden_size) or not torch.isfinite(q).all():
            raise ValueError("Invalid rotation shape or values")
        error = q.T @ q - torch.eye(q.shape[0], device=device, dtype=compute_dtype)
        maximum = error.abs().max().item()
        if maximum > orthogonality_tol:
            raise ValueError(f"Rotation orthogonality error {maximum:g} exceeds {orthogonality_tol:g}")
        diagnostics.append({"max_abs_RtR_minus_I": maximum,
                            "relative_frobenius_RtR_minus_I": error.norm().item() / q.shape[0] ** .5})

    def rotate_weight(module, r, output=False):
        q = r.to(device=device, dtype=compute_dtype)
        w = module.weight.detach().to(device=device, dtype=compute_dtype)
        rotated = q.T @ w if output else w @ q
        module.weight.copy_(rotated.to(device=module.weight.device, dtype=module.weight.dtype))
        if output and getattr(module, "bias", None) is not None:
            b = q.T @ module.bias.detach().to(device=device, dtype=compute_dtype)
            module.bias.copy_(b.to(device=module.bias.device, dtype=module.bias.dtype))

    def transition(a, b, reference):
        if a is b:
            return None
        value = a.to(device=device, dtype=compute_dtype).T @ b.to(device=device, dtype=compute_dtype)
        return value.to(device=reference.device, dtype=reference.dtype)

    rotate_weight(model.model.embed_tokens, rotations[0])
    for i, layer in enumerate(layers):
        print(f"[R1 capture] Fuse rotations: layer {i}", flush=True)
        a, b, nxt = rotations[2 * i:2 * i + 3]
        for module in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            rotate_weight(module, a)
        rotate_weight(layer.self_attn.o_proj, b, output=True)
        for module in (layer.mlp.gate_proj, layer.mlp.up_proj):
            rotate_weight(module, b)
        rotate_weight(layer.mlp.down_proj, nxt, output=True)
        layer.attn_residual_rotation = transition(a, b, layer.self_attn.q_proj.weight)
        layer.mlp_residual_rotation = transition(b, nxt, layer.self_attn.q_proj.weight)
    rotate_weight(model.lm_head, rotations[-1])
    return diagnostics


@torch.no_grad()
def capture_r1_inputs(model, input_ids, output_dir, mode, metadata=None):
    if mode not in ("norm_fused", "global", "layerwise"):
        raise ValueError("Unknown R1 capture mode")
    rotated = mode != "norm_fused"
    validate_fp_model(model, allow_residual_rotations=rotated)
    norms = [model.model.norm] + [n for l in model.model.layers for n in (l.input_layernorm, l.post_attention_layernorm)]
    if any(not torch.equal(n.weight, torch.ones_like(n.weight)) for n in norms):
        raise ValueError("Expected unit norm scales")
    return _capture_projection_inputs(model, input_ids, output_dir, metadata, profile={
        "capture_mode": "fp_r1_single_sequence", "rotation_layout": mode,
        "rotate": rotated, "norm_fused": True, "embedding_centered": False,
        "r1_applied": rotated, "r2_applied": False, "r3_applied": False, "r4_applied": False,
        "activation_basis": "post-RMSNorm without gamma" + (", rotated by R1" if rotated else ", original channels"),
    })


def distribution_metrics(x):
    """All-token metrics. N80/entropy use channel squared-energy probabilities."""
    v = x.float().reshape(-1, x.shape[-1])
    if not torch.isfinite(v).all():
        raise ValueError("Nonfinite activations")
    energy = v.square().sum(0)
    total = energy.sum()
    p = energy / total if total > 0 else torch.zeros_like(energy)
    nonzero = p[p > 0]
    sorted_p = p.sort(descending=True).values
    token_energy = v.square().sum(1)
    peaks = v.abs().amax(1)
    kappa = torch.where(token_energy > 0, v.shape[1] * peaks.square() / token_energy.clamp_min(1e-30), 0.)
    return {
        "max_abs": peaks.max().item(), "rms": v.square().mean().sqrt().item(),
        "channel_energy_entropy_normalized": (-(nonzero * nonzero.log()).sum() / torch.tensor(float(v.shape[1])).log()).item() if v.shape[1] > 1 else 0.,
        "channel_energy_n80": min(int(torch.searchsorted(sorted_p.cumsum(0), .8)) + 1, v.shape[1]) if total > 0 else 0,
        "top8_channel_energy_fraction": sorted_p[:8].sum().item(),
        "token_kappa_mean": kappa.mean().item(), "token_kappa_p99": torch.quantile(kappa, .99).item(),
        "token0_max_abs": peaks[0].item(), "token0_kappa": kappa[0].item(),
    }


def difference_metrics(actual, expected):
    a, b = actual.float(), expected.float()
    delta = a - b
    return {"max_abs_difference": delta.abs().max().item(),
            "relative_l2_error": (delta.norm() / b.norm().clamp_min(1e-30)).item()}
