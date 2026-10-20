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


def validate_group_boundaries(group_boundaries, num_layers=None):
    """Validate zero-based block indices AFTER which a new group starts.

    [] is one group. Boundaries must already be strictly increasing; never
    silently reorder a layout supplied by the caller or a checkpoint.
    """
    if not isinstance(group_boundaries, list):
        raise ValueError("group_boundaries must be a list of integers")
    if any(type(b) is not int or b < 0 for b in group_boundaries):
        raise ValueError("group_boundaries must contain nonnegative integers")
    if any(a >= b for a, b in zip(group_boundaries, group_boundaries[1:])):
        raise ValueError("group_boundaries must be strictly increasing with no duplicates")
    if num_layers is not None:
        _positive_integer(num_layers, "num_layers")
        if any(b >= 2 * num_layers - 1 for b in group_boundaries):
            raise ValueError(f"group_boundaries must be in [0, {2 * num_layers - 2}]; "
                             "the final output shares the last group")


def rotation_group_indices(num_layers, groupnum=None, group_boundaries=None):
    """Return 2*num_layers + 1 group IDs, including the final output.

    groupnum selects equal-sized consecutive groups; group_boundaries selects
    explicit splits after the given block indices. Supply exactly one.
    For example, two layers and two groups give [0, 0, 1, 1, 1].
    """
    _positive_integer(num_layers, "num_layers")
    if group_boundaries is not None:
        if groupnum is not None:
            raise ValueError("Specify only one of groupnum or group_boundaries")
        validate_group_boundaries(group_boundaries, num_layers)
        boundaries = set(group_boundaries)
        ids, group = [], 0
        for block in range(2 * num_layers):
            ids.append(group)
            if block in boundaries:
                group += 1
        return ids + [group]
    _positive_integer(groupnum, "groupnum")
    num_blocks = 2 * num_layers
    if num_blocks % groupnum:
        raise ValueError(
            f"groupnum ({groupnum}) must divide 2 * num_layers ({num_blocks})"
        )
    blocks_per_group = num_blocks // groupnum
    return [block // blocks_per_group for block in range(num_blocks)] + [groupnum - 1]


def rotation_group_metadata(num_layers, groupnum=None, group_boundaries=None):
    """Describe the layout using values safe for torch.load(weights_only=True)."""
    group_ids = rotation_group_indices(num_layers, groupnum, group_boundaries)
    metadata = {
        "version": 1 if group_boundaries is None else 2,
        "num_hidden_layers": num_layers,
        "groupnum": group_ids[-1] + 1,
        "group_indices": group_ids,
    }
    if group_boundaries is not None:
        metadata["group_boundaries"] = list(group_boundaries)
    return metadata


def checkpoint_group_indices(checkpoint, num_layers, groupnum=None, layerwise=False,
                             group_boundaries=None):
    """Resolve and validate the evaluation layout without loading any tensors.

    Return the exact group IDs for grouped rotations, or None for legacy R1/A/B
    layouts. Grouped checkpoints must carry metadata; legacy A/B checkpoints
    require --layerwise. A legacy R1 checkpoint also supports --groupnum 1
    or an explicitly empty boundary list.
    R2 and tensor shapes are handled by the evaluation caller.
    """
    _positive_integer(num_layers, "num_layers")
    requested_ids = None
    if groupnum is not None or group_boundaries is not None:
        requested_ids = rotation_group_indices(num_layers, groupnum, group_boundaries)
        if layerwise:
            raise ValueError("Grouped rotations cannot be combined with --layerwise")
    if checkpoint is None:
        return requested_ids
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
        if type(metadata.get("version")) is not int or metadata["version"] not in (1, 2):
            raise ValueError("Unsupported rotation metadata version")
        saved_layers = metadata.get("num_hidden_layers")
        _positive_integer(saved_layers, "checkpoint num_hidden_layers")
        if saved_layers != num_layers:
            raise ValueError(
                f"Checkpoint has {saved_layers} layers, but model has {num_layers}"
            )
        saved_groupnum = metadata.get("groupnum")
        _positive_integer(saved_groupnum, "checkpoint groupnum")
        if metadata["version"] == 2:
            validate_group_boundaries(metadata.get("group_boundaries"), num_layers)
            expected_ids = rotation_group_indices(num_layers, group_boundaries=metadata["group_boundaries"])
            if saved_groupnum != expected_ids[-1] + 1:
                raise ValueError("Checkpoint groupnum does not match its group_boundaries")
        else:
            if "group_boundaries" in metadata:
                raise ValueError("Custom boundaries require rotation metadata version 2")
            expected_ids = rotation_group_indices(num_layers, saved_groupnum)
        if metadata.get("group_indices") != expected_ids:
            raise ValueError("Checkpoint group_indices do not match the saved rotation layout")
        if groupnum is not None and groupnum != saved_groupnum:
            raise ValueError(
                f"Requested groupnum {groupnum} does not match checkpoint groupnum {saved_groupnum}"
            )
        if requested_ids is not None and requested_ids != expected_ids:
            raise ValueError("Requested rotation boundaries do not match the checkpoint layout")
        if "R1" in checkpoint or ab_keys:
            raise ValueError("Grouped checkpoints cannot contain legacy R1/A/B rotations")
        if q_keys != {f"Q.{index}" for index in range(saved_groupnum)}:
            raise ValueError("Checkpoint must contain exactly Q.0 through Q.(groupnum - 1)")
        return expected_ids

    if q_keys:
        raise ValueError(f"Grouped Q rotations require {METADATA_KEY}")
    if ab_keys:
        if "R1" in checkpoint:
            raise ValueError("Checkpoint mixes legacy R1 and A/B rotations")
        if requested_ids is not None or not layerwise:
            raise ValueError("Legacy A/B checkpoints require --layerwise and no grouped layout")
        expected_keys = {f"A.{index}" for index in range(num_layers + 1)}
        expected_keys.update(f"B.{index}" for index in range(num_layers))
        if ab_keys != expected_keys:
            raise ValueError("Legacy checkpoint must contain A.0 through A.L and B.0 through B.(L - 1)")
        return None
    if "R1" in checkpoint:
        if layerwise or (requested_ids is not None and requested_ids[-1] != 0):
            raise ValueError("Legacy R1 checkpoints support only global rotation (--groupnum 1)")
        return requested_ids
    raise ValueError("Checkpoint contains no R1, A/B, or grouped Q rotations")


def checkpoint_groupnum(checkpoint, num_layers, groupnum=None, layerwise=False,
                        group_boundaries=None):
    """Compatibility helper; use checkpoint_group_indices to reconstruct layouts."""
    ids = checkpoint_group_indices(checkpoint, num_layers, groupnum, layerwise, group_boundaries)
    return ids[-1] + 1 if ids is not None else None
