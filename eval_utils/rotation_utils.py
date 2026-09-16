# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import functools
import math

import torch
import tqdm

from utils import monkeypatch, quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    hadamard_matrix,
    is_pow2,
    random_hadamard_matrix,
)
from utils.utils import HadamardTransform


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1, r4=True):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if r4:
        apply_exact_had_to_linear(
            W, had_dim=-1, output=False
        )  # apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, R2=None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)
    if v_proj.bias is not None:
        rotation = R2 if R2 is not None else hadamard_matrix(head_dim, v_proj.bias.device)
        bias = v_proj.bias.data
        v_proj.bias.data = (
            bias.reshape(-1, head_dim).double() @ rotation.to(device=bias.device, dtype=torch.float64)
        ).reshape_as(bias).to(bias.dtype)


@torch.no_grad()
def rotate_model(model, args):
    checkpoint = (
        torch.load(args.optimized_rotation_path, map_location="cpu", weights_only=True)
        if args.optimized_rotation_path is not None else None
    )
    respin = getattr(args, "respin", False) or (checkpoint is not None and "A.0" in checkpoint)
    if respin and getattr(args, "export_to_et", False):
        raise ValueError("ExecuTorch export does not support Respin residual rotations.")
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = getattr(config, "head_dim", model_dim // num_heads)
    num_layers = len(model.model.layers)

    if checkpoint is not None:
        sizes = {f"model.layers.{i}.self_attn.R2": head_dim for i in range(num_layers)}
        if respin:
            sizes.update({f"A.{i}": model_dim for i in range(num_layers + 1)})
            sizes.update({f"B.{i}": model_dim for i in range(num_layers)})
        else:
            sizes["R1"] = model_dim
        for key, size in sizes.items():
            if key not in checkpoint or checkpoint[key].shape != (size, size):
                raise ValueError(f"Rotation checkpoint requires {key} with shape {(size, size)}.")

    def rotation(key, size):
        if checkpoint is not None:
            return checkpoint[key].to(device="cuda", dtype=torch.float64)
        return get_orthogonal_matrix(size, args.rotate_mode)

    config.respin = respin
    A = rotation("A.0" if respin else "R1", model_dim)

    rotate_embeddings(model, A)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        B = rotation(f"B.{idx}", model_dim) if respin else A
        A_next = rotation(f"A.{idx + 1}", model_dim) if respin else A
        R2 = rotation(f"model.layers.{idx}.self_attn.R2", head_dim)
        rotate_attention_inputs(layer, A)
        rotate_attention_output(layer, B)
        rotate_mlp_input(layer, B)
        rotate_mlp_output(layer, A_next, r4=args.r4)
        rotate_ov_proj(layer, num_heads, head_dim, R2=R2)
        if respin:
            dtype = layer.self_attn.q_proj.weight.dtype
            layer.attn_residual_rotation = (A.T @ B).to(device="cpu", dtype=dtype)
            layer.mlp_residual_rotation = (B.T @ A_next).to(device="cpu", dtype=dtype)
        A = A_next
    rotate_head(model, A)


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        self.r3 = kwargs.get("r3", True)
        if self.r3:
            assert is_pow2(head_dim), "R3 requires a power-of-two head_dim!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_bits"] >= 16 or kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        if self.r3:
            q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
            k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        if self.k_bits >= 16:
            return q, k
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.reshape(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)
