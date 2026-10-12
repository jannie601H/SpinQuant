# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm

from utils import model_utils
from utils.activation_stats import LayerActivationStats


@torch.no_grad()
def evaluator(model, testenc, dev, args, stats_metadata=None):
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)

    layers[0] = layers[0].to(dev)

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    if model.seqlen < 2:
        raise ValueError("Evaluation sequence length must be at least two")
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    if nsamples == 0:
        raise ValueError("Evaluation requires at least one full sequence")
    input_ids = (
        input_ids[:, : nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)
    )  # (nsamples, seqlen)

    capture_stats = getattr(args, "capture_activation_stats", False)
    if torch.distributed.is_initialized():
        capture_stats = capture_stats and torch.distributed.get_rank() == 0

    stats_dir = None
    if capture_stats:
        if args.a_bits >= 16:
            raise ValueError("Activation statistics require a_bits < 16")
        if args.capture_layer_io:
            raise ValueError("Disable capture_layer_io when collecting activation statistics")
        if not args.activation_stats_dir:
            raise ValueError("Specify activation_stats_dir when collecting activation statistics")

        stats_dir = Path(args.activation_stats_dir)
        # Reserve a fresh directory so partial/previous collections cannot mix.
        stats_dir.mkdir(parents=True, exist_ok=False)
        torch.save({"input_ids": input_ids.detach().cpu()}, stats_dir / "tokens.pt")
        metadata = {
            **(stats_metadata or {}),
            "schema_version": 1,
            "dataset": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "test",
            "num_layers": len(layers),
            "num_sequences": nsamples,
            "sequence_length": model.seqlen,
            "evaluation_batch_size": args.bsz,
            "seed": args.seed,
            "layerwise": args.layerwise,
            "rotate": args.rotate,
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "sites": {
                "attn": "self_attn.q_proj.quantizer",
                "ffn": "mlp.gate_proj.quantizer",
            },
            "sequence_order": "same row order as tokens.pt",
            "error_definition": "L2(x.float() - x_hat.float()) over channels",
            "statistics_dtype": "float32",
            "evaluation_time_includes_collection": True,
            "status": "collecting",
        }
        (stats_dir / "manifest.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    batch_size = args.bsz
    input_ids = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    # Catcher fills each entry with the actual batch (including a short last batch).
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    position_ids = cache["position_ids"]

    torch.cuda.empty_cache()
    outs = [0] * nbatches
    attention_mask = cache["attention_mask"]

    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)

        # Dump the layer input and output
        if args.capture_layer_io and args.layer_idx == i:
            captured_io = model_utils.capture_layer_io(layer, inps)
            save_path = model_utils.get_layer_io_save_path(args)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(captured_io, save_path)
            logging.info(f"Dumped layer input and output to: {save_path}")

        stats = LayerActivationStats(i, stats_dir) if capture_stats else None
        hook_context = stats.attach(layer) if stats is not None else nullcontext()
        with hook_context:
            for j in range(nbatches):
                if stats is not None:
                    stats.begin_batch(j, inps[j].shape[0], inps[j].shape[1])

                outs[j] = layer(
                    inps[j],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

                if stats is not None:
                    stats.end_batch()

        if stats is not None:
            stats.save()
        del stats, hook_context
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # if model.model.norm is not None:
    #     model.model.norm = model.model.norm.to(dev)

    # model.lm_head = model.lm_head.to(dev)
    # nlls = []
    # loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # for i in range(nbatches):
    #     hidden_states = inps[i]
    #     if model.model.norm is not None:
    #         hidden_states = model.model.norm(hidden_states)
    #     lm_logits = model.lm_head(hidden_states)
    #     shift_logits = lm_logits[:, :-1, :]
    #     shift_labels = input_ids[i][:, 1:]
    #     loss = loss_fct(shift_logits.float().permute(0, 2, 1), shift_labels)
    #     neg_log_likelihood = loss.float().mean(dim=1)
    #     nlls.append(neg_log_likelihood)
    # nlls_tensor = torch.cat(nlls)
    # ppl = torch.exp(nlls_tensor.mean())

    # Decoder 계산에서 사용했던 이전 layer 출력은 이제 필요 없음
    del outs
    torch.cuda.empty_cache()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)

    model.lm_head = model.lm_head.to(dev)

    token_chunk_size = 128
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    total_nll = torch.zeros((), device=dev, dtype=torch.float64)
    total_tokens = 0

    for i in tqdm(range(nbatches), desc="(Eval) PPL"):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)

        # 이 배치의 입력을 리스트에서 해제
        inps[i] = None
        seq_len = hidden_states.shape[1]

        for start in range(0, seq_len - 1, token_chunk_size):
            end = min(start + token_chunk_size, seq_len - 1)

            # 전체 sequence가 아닌 작은 구간만 vocabulary로 projection
            logits = model.lm_head(hidden_states[:, start:end, :])
            labels = input_ids[i][:, start + 1:end + 1]

            # loss는 FP32로 계산하고, 합계는 FP64로 누적
            chunk_nll = loss_fct(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )
            total_nll += chunk_nll.double()
            total_tokens += labels.numel()

            del logits, labels, chunk_nll

        del hidden_states

    ppl = torch.exp(total_nll / total_tokens)
    if capture_stats:
        metadata["status"] = "complete"
        metadata["ppl"] = ppl.item()
        (stats_dir / "manifest.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logging.info("Saved activation statistics to: %s", stats_dir)
    model.config.use_cache = use_cache
    logging.info(f"\n WikiText2 PPL: {ppl.item():.3f}")
    return ppl.item()
