from contextlib import contextmanager
from pathlib import Path

import torch


class LayerActivationStats:
    """Collect input-quantization statistics for one decoder layer."""

    def __init__(self, layer_index, output_dir):
        self.layer_index = layer_index
        self.output_dir = Path(output_dir)

        self.records = {
            "attn": [],
            "ffn": [],
        }
        self.quantizer_config = {}

        self.batch_shape = None
        self.seen_sites = set()
        self.next_batch_index = 0

    def begin_batch(self, batch_index, batch_size, sequence_length):
        if self.batch_shape is not None:
            raise RuntimeError("Previous activation-stat batch is unfinished")
        if batch_index != self.next_batch_index:
            raise RuntimeError("Activation-stat batches must be sequential")

        self.batch_shape = (batch_size, sequence_length)
        self.seen_sites.clear()

    def end_batch(self):
        if self.batch_shape is None:
            raise RuntimeError("No activation-stat batch is active")
        if self.seen_sites != {"attn", "ffn"}:
            raise RuntimeError(
                f"Layer {self.layer_index}: missing quantizer calls; "
                f"observed={sorted(self.seen_sites)}"
            )

        self.next_batch_index += 1
        self.batch_shape = None

    def make_hook(self, site):
        @torch.no_grad()
        def hook(quantizer, inputs, output):
            if self.batch_shape is None:
                raise RuntimeError("begin_batch() must run before the layer")

            if site in self.seen_sites:
                raise RuntimeError(
                    f"Layer {self.layer_index}: duplicate call for {site}"
                )

            # read quantizer input, output, scale
            x = inputs[0].detach()
            x_hat = output.detach()
            scale = quantizer.scale

            if scale is None:
                raise RuntimeError("Quantizer scale was freed before capture")

            if x.ndim != 3 or tuple(x.shape[:2]) != self.batch_shape:
                raise RuntimeError(
                    f"Unexpected activation shape: {tuple(x.shape)}"
                )

            if x_hat.shape != x.shape or scale.shape != x.shape:
                raise RuntimeError("Activation/scale shapes do not match")

            x_fp32 = x.float()
            x_hat_fp32 = x_hat.float()

            x_norm = torch.linalg.vector_norm(x_fp32, ord=2, dim=-1)
            error_norm = torch.linalg.vector_norm(
                x_fp32 - x_hat_fp32, ord=2, dim=-1
            )

            if quantizer.groupsize > 0:
                groupsize = quantizer.groupsize
                if x.shape[-1] % groupsize != 0:
                    raise RuntimeError("Invalid activation groupsize")

                compact_scale = scale.detach().reshape(
                    *x.shape[:-1],
                    x.shape[-1] // groupsize,
                    groupsize,
                )[..., 0]
            else:
                compact_scale = scale.detach()[..., 0]

            self.records[site].append({
                "scale": compact_scale.to(
                    device="cpu", dtype=torch.float32
                ).clone(),
                "x_norm": x_norm.cpu(),
                "error_norm": error_norm.cpu(),
            })

            self.seen_sites.add(site)
            # A forward hook returning None leaves the model output unchanged.

        return hook

    @contextmanager
    def attach(self, layer):
        targets = {
            "attn": layer.self_attn.q_proj.quantizer,
            "ffn": layer.mlp.gate_proj.quantizer,
        }

        handles = []
        try:
            for site, quantizer in targets.items():
                if quantizer.bits >= 16:
                    raise ValueError(
                        f"{site}: activation quantization must be enabled"
                    )

                self.quantizer_config[site] = {
                    "bits": quantizer.bits,
                    "groupsize": quantizer.groupsize,
                    "sym": quantizer.sym,
                    "clip_ratio": quantizer.clip_ratio,
                }

                handles.append(
                    quantizer.register_forward_hook(self.make_hook(site))
                )

            yield self
        finally:
            for handle in handles:
                handle.remove()

    def save(self):
        if self.batch_shape is not None:
            raise RuntimeError("Cannot save an unfinished batch")

        for site, records in self.records.items():
            if not records:
                raise RuntimeError(
                    f"Layer {self.layer_index}: no records for {site}"
                )

            tensors = {
                name: torch.cat(
                    [record[name] for record in records], dim=0
                )
                for name in ("scale", "x_norm", "error_norm")
            }

            payload = {
                "layer_index": self.layer_index,
                "site": site,
                "quantizer": self.quantizer_config[site],
                "norm": "L2 over the last activation dimension, in FP32",
                "num_sequences": tensors["x_norm"].shape[0],
                **tensors,
            }

            path = self.output_dir / (
                f"layer_{self.layer_index:03d}_{site}.pt"
            )
            torch.save(payload, path)
            records.clear()
