# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared rotation layouts for training and checkpoint-based evaluation.

Block inputs are ordered A.0, B.0, A.1, B.1, ... (attention, then MLP).
The extra final output position uses the last block's group.
"""

from collections.abc import Mapping


METADATA_KEY = "_rotation_metadata"


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def rotation_group_indices(num_layers, groupnum):
    """Return 2*num_layers + 1 group IDs, including the final output.

    Groups contain equal numbers of consecutive Attention/MLP blocks.
    For example, two layers and two groups give [0, 0, 1, 1, 1].
    """
    _positive_integer(num_layers, "num_layers")
    _positive_integer(groupnum, "groupnum")
    num_blocks = 2 * num_layers
    if num_blocks % groupnum:
        raise ValueError(
            f"groupnum ({groupnum}) must divide 2 * num_layers ({num_blocks})"
        )
    blocks_per_group = num_blocks // groupnum
    return [block // blocks_per_group for block in range(num_blocks)] + [groupnum - 1]


def rotation_group_metadata(num_layers, groupnum):
    """Describe the layout using values safe for torch.load(weights_only=True)."""
    group_ids = rotation_group_indices(num_layers, groupnum)
    return {
        "version": 1,
        "num_hidden_layers": num_layers,
        "groupnum": groupnum,
        "group_indices": group_ids,
    }


def checkpoint_groupnum(checkpoint, num_layers, groupnum=None, layerwise=False):
    """Resolve and validate the evaluation layout without loading any tensors.

    Return a group count for grouped rotations, or None for legacy R1/A/B
    layouts. Grouped checkpoints must carry metadata; legacy A/B checkpoints
    require --layerwise. A legacy R1 checkpoint also supports --groupnum 1.
    R2 and tensor shapes are handled by the evaluation caller.
    """
    _positive_integer(num_layers, "num_layers")
    if groupnum is not None:
        rotation_group_indices(num_layers, groupnum)
        if layerwise:
            raise ValueError("--groupnum cannot be combined with --layerwise")
    if checkpoint is None:
        return groupnum
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Rotation checkpoint must be a mapping")

    q_keys = {key for key in checkpoint if isinstance(key, str) and key.startswith("Q.")}
    ab_keys = {
        key for key in checkpoint
        if isinstance(key, str) and key.startswith(("A.", "B."))
    }
    if METADATA_KEY in checkpoint:
        if layerwise:
            raise ValueError("Grouped rotation checkpoints cannot be used with --layerwise")
        metadata = checkpoint[METADATA_KEY]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{METADATA_KEY} must be a mapping")
        if type(metadata.get("version")) is not int or metadata["version"] != 1:
            raise ValueError("Unsupported rotation metadata version")
        saved_layers = metadata.get("num_hidden_layers")
        _positive_integer(saved_layers, "checkpoint num_hidden_layers")
        if saved_layers != num_layers:
            raise ValueError(
                f"Checkpoint has {saved_layers} layers, but model has {num_layers}"
            )
        saved_groupnum = metadata.get("groupnum")
        expected_ids = rotation_group_indices(num_layers, saved_groupnum)
        if metadata.get("group_indices") != expected_ids:
            raise ValueError("Checkpoint group_indices do not match the uniform rotation layout")
        if groupnum is not None and groupnum != saved_groupnum:
            raise ValueError(
                f"Requested groupnum {groupnum} does not match checkpoint groupnum {saved_groupnum}"
            )
        if "R1" in checkpoint or ab_keys:
            raise ValueError("Grouped checkpoints cannot contain legacy R1/A/B rotations")
        if q_keys != {f"Q.{index}" for index in range(saved_groupnum)}:
            raise ValueError("Checkpoint must contain exactly Q.0 through Q.(groupnum - 1)")
        return saved_groupnum

    if q_keys:
        raise ValueError(f"Grouped Q rotations require {METADATA_KEY}")
    if ab_keys:
        if "R1" in checkpoint:
            raise ValueError("Checkpoint mixes legacy R1 and A/B rotations")
        if groupnum is not None or not layerwise:
            raise ValueError("Legacy A/B checkpoints require --layerwise and no --groupnum")
        expected_keys = {f"A.{index}" for index in range(num_layers + 1)}
        expected_keys.update(f"B.{index}" for index in range(num_layers))
        if ab_keys != expected_keys:
            raise ValueError("Legacy checkpoint must contain A.0 through A.L and B.0 through B.(L - 1)")
        return None
    if "R1" in checkpoint:
        if layerwise or groupnum not in (None, 1):
            raise ValueError("Legacy R1 checkpoints support only global rotation (--groupnum 1)")
        return groupnum
    raise ValueError("Checkpoint contains no R1, A/B, or grouped Q rotations")
