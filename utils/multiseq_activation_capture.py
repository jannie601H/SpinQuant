"""Post-fusion FP inputs from disjoint, reproducible calibration block splits."""

import hashlib
import json
from pathlib import Path
import random

import torch

from utils.fp_activation_capture import validate_fp_model, _capture_projection_inputs
from utils.fuse_norm_utils import fuse_layer_norms


CAPTURE_MODE = "fp_fused_multisequence"


def select_blocks(blocks, sequence_length=2048, seed=0, train_count=64, test_count=32):
    """Sample without replacement from CustomJsonDataset.input_ids (train split).

    Incomplete trailing blocks are excluded. Local RNG leaves training RNG alone.
    Returns only selected CPU tokens, in train-then-test order, and provenance.
    """
    for name, value in (("sequence_length", sequence_length), ("train_count", train_count),
                        ("test_count", test_count)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    eligible = [i for i, block in enumerate(blocks) if len(block) == sequence_length]
    total = train_count + test_count
    if len(eligible) < total:
        raise ValueError(f"Need {total} complete blocks, found {len(eligible)}")
    indices = random.Random(seed).sample(eligible, total)
    tokens = torch.stack([torch.as_tensor(blocks[i], dtype=torch.long).cpu() for i in indices])
    records = []
    for index, source in enumerate(indices):
        split = "train" if index < train_count else "test"
        split_index = index if split == "train" else index - train_count
        records.append({
            "activation_sequence_index": index, "grouping_split": split,
            "split_sequence_index": split_index, "source_sequence_index": source,
            "source_token_offset": source * sequence_length,
            "relative_dir": f"{split}/sequence_{split_index:03d}",
            "token_sha256": hashlib.sha256(tokens[index].numpy().tobytes()).hexdigest(),
        })
    return tokens, {
        "seed": seed, "sequence_length": sequence_length,
        "num_sequences": total, "split_counts": {"train": train_count, "test": test_count},
        "source_num_blocks": len(blocks), "source_complete_blocks": len(eligible),
        "sequence_selection": "seeded sample without replacement of complete nonoverlapping blocks",
        "token_hash_encoding": "CPU int64 native-endian bytes",
        "sequences": records,
    }


def _validate_selection(tokens, selection):
    records = selection["sequences"]
    counts = selection["split_counts"]
    if tokens.dtype != torch.long or tokens.device.type != "cpu":
        raise ValueError("Selected tokens must be CPU int64")
    if tuple(tokens.shape) != (len(records), selection["sequence_length"]):
        raise ValueError("Selected token shape does not match selection")
    if counts.keys() != {"train", "test"} or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("Expected nonempty train/test splits")
    if sum(counts.values()) != len(records) or selection["num_sequences"] != len(records):
        raise ValueError("Split counts do not match selection")
    if len({r["source_sequence_index"] for r in records}) != len(records):
        raise ValueError("Source blocks overlap between selected sequences")
    for i, record in enumerate(records):
        split = "train" if i < counts["train"] else "test"
        local = i if split == "train" else i - counts["train"]
        if (record["activation_sequence_index"] != i or record["grouping_split"] != split
                or record["split_sequence_index"] != local
                or record["relative_dir"] != f"{split}/sequence_{local:03d}"
                or record["token_sha256"] != hashlib.sha256(tokens[i].numpy().tobytes()).hexdigest()):
            raise ValueError("Selection metadata does not match tokens/order")


@torch.no_grad()
def capture_fused_sequences(model, tokens, selection, output_dir, metadata=None):
    """Fuse a fresh original model in-place once, then capture independent forwards.

    train/test are grouping roles, both sourced from WikiText-2 train; neither
    means the official evaluation split. No rotation, quantization or LM head run.
    """
    _validate_selection(tokens, selection)
    validate_fp_model(model)
    if getattr(model, "_multiseq_norm_fused", False):
        raise ValueError("Use a fresh original model; fusion was already applied")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        **(metadata or {}), **selection,
        "schema_version": 2, "capture_mode": CAPTURE_MODE, "status": "preparing",
        "dataset": "Salesforce/wikitext", "subset": "wikitext-2-raw-v1",
        "source_split": "train", "split": "train",
        "split_roles": {"train": "grouping construction", "test": "grouping holdout"},
        "tokenization": "CustomJsonDataset: tokenize each text, concatenate without added separators, drop incomplete block",
        "rotate": False, "norm_fused": False, "embedding_centered": False,
        "qk_hadamard": False, "w_bits": 16, "a_bits": 16, "k_bits": 16, "v_bits": 16,
        "activation_basis": "original channel coordinates, post-RMSNorm with gamma fused into weights",
        "num_layers": len(model.model.layers), "num_tokens": tokens.numel(),
        "activation_dtype": str(model.model.embed_tokens.weight.dtype),
        "model_device": str(model.model.embed_tokens.weight.device),
        "statistics_dtype": "torch.float32", "statistics_scope": "per sequence; raw x retained for pooled statistics",
        "sites": {"attn": "self_attn.q_proj input", "ffn": "mlp.gate_proj input"},
        "sequence_order": "independent forward per sequence; train first, then test",
        "padding": False, "use_cache": False, "ppl_evaluated": False,
        "calibration_performed": False, "completed_sequences": 0,
    }

    def save_manifest():
        temporary = directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(directory / "manifest.json")

    save_manifest()
    was_training = model.training
    try:
        # Match optimize_rotation.py: final-norm fusion must not mutate shared embeddings.
        tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        if tied:
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings = False
        model.eval()
        fuse_layer_norms(model)
        model._multiseq_norm_fused = True
        norms = [model.model.norm] + [n for layer in model.model.layers
                  for n in (layer.input_layernorm, layer.post_attention_layernorm)]
        if any(not torch.equal(n.weight, torch.ones_like(n.weight)) for n in norms):
            raise RuntimeError("Norm fusion left non-unit RMSNorm weights")
        manifest.update(status="collecting", norm_fused=True, embedding_centered=True,
                        head_was_tied=tied, fusion_function="utils.fuse_norm_utils.fuse_layer_norms")
        save_manifest()
        start = 0
        for split in ("train", "test"):
            count = selection["split_counts"][split]
            split_dir = directory / split
            split_dir.mkdir()
            torch.save({"input_ids": tokens[start:start + count].clone(),
                        "source_sequence_indices": [r["source_sequence_index"]
                            for r in selection["sequences"][start:start + count]],
                        "source_split": "train", "grouping_split": split}, split_dir / "tokens.pt")
            start += count
        for i, record in enumerate(selection["sequences"]):
            print(f"[Fused FP capture] {i + 1}/{len(tokens)} {record['relative_dir']} "
                  f"source_block={record['source_sequence_index']}", flush=True)
            _capture_projection_inputs(model, tokens[i:i + 1], directory / record["relative_dir"],
                metadata={**(metadata or {}), **record, "source_split": "train", "split": "train"},
                profile={"schema_version": 2, "capture_mode": CAPTURE_MODE,
                         "norm_fused": True, "embedding_centered": True,
                         "activation_basis": manifest["activation_basis"],
                         "activation_sequence_index": i})
            manifest["completed_sequences"] = i + 1
            save_manifest()
        manifest["status"] = "complete"
        save_manifest()
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        save_manifest()
        raise
    finally:
        model.train(was_training)
    return manifest
