"""Run rotation optimization only when its completed, matching cache is absent."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def option(command, flag, default=None):
    return command[command.index(flag) + 1] if flag in command else default


def rotation_config(command):
    """Only optimization settings belong to a checkpoint shared by PTQ targets."""
    return {
        "model": option(command, "--input_model"),
        "rotation_w_bits": int(option(command, "--w_bits")),
        **{f"{name}_bits": int(option(command, f"--{name}_bits")) for name in "akv"},
        "rotation": True,
        "had": "--r3" in command or "--r4" in command,
        "r3": "--r3" in command,
        "r4": "--r4" in command,
        "max_steps": int(option(command, "--max_steps")),
        "learning_rate": float(option(command, "--learning_rate")),
        "seed": int(option(command, "--seed", "0")),
    }


def cache_spec(command):
    command = list(command)
    model_index = command.index("--input_model") + 1
    model = Path(command[model_index]).expanduser()
    local_files = None
    if model.is_dir():
        model = model.resolve()
        command[model_index] = str(model)
        # Avoid reading multi-GB weights on every cache lookup. Local model
        # replacement is detected by relative filename, size and mtime.
        local_files = {
            str(path.relative_to(model)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in sorted(model.rglob("*")) if path.is_file()
        }
    sources = [REPO_ROOT / "optimize_rotation.py"]
    for directory in ("train_utils", "utils"):
        sources.extend(sorted((REPO_ROOT / directory).glob("*.py")))
    packages = {}
    for name in ("torch", "transformers", "accelerate", "datasets", "fast-hadamard-transform"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema": 1,
        "command": command,
        "local_model_files": local_files,
        "training_sources": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in sources},
        "packages": packages,
    }


def cache_label(command):
    def value(flag):
        return command[command.index(flag) + 1]

    model = Path(value("--input_model")).name
    bits = "".join(f"{name.upper()}{value('--' + name + '_bits')}" for name in "wakv")
    r3 = "off" if "--no-r3" in command else "on"
    r4 = "off" if "--no-r4" in command else "on"
    label = f"{model}_{bits}_steps{value('--max_steps')}_r3-{r3}_r4-{r4}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", label)[:160]


def valid_cache(directory, spec):
    checkpoint = directory / "R.bin"
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
        return (
            metadata["spec"] == spec
            and checkpoint.stat().st_size > 0
            and metadata["checkpoint_sha256"] == sha256(checkpoint)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ensure_rotation(cache_root, command, force=False):
    if "--output_rotation_path" in command or "--output_dir" in command:
        raise ValueError("Output directories are managed by the rotation cache")
    spec = cache_spec(command)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(cache_root).resolve() / f"{cache_label(command)}_{key}"
    directory.mkdir(parents=True, exist_ok=True)
    # Serialize optimization for the same configuration across shell invocations.
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not force and valid_cache(directory, spec):
            print(f"Reusing rotation: {directory / 'R.bin'}", file=sys.stderr)
            return directory / "R.bin"

        print(f"Optimizing rotation: {directory}", file=sys.stderr)
        # Failed/interrupted training never publishes a reusable checkpoint.
        # An existing completed checkpoint survives a failed forced retraining.
        with tempfile.TemporaryDirectory(prefix="training-", dir=directory) as staging:
            staging = Path(staging)
            run_command = spec["command"] + [
                "--output_rotation_path", str(staging),
                "--output_dir", str(staging / "trainer"),
            ]
            with (directory / "optimize.log").open("w") as log:
                with subprocess.Popen(
                    run_command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True,
                ) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        sys.stderr.write(line)
                    returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, run_command)
            checkpoint = staging / "R.bin"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise RuntimeError("Optimization exited without a nonempty R.bin")
            metadata = {
                "spec": spec,
                "checkpoint_sha256": sha256(checkpoint),
                "rotation_config": rotation_config(spec["command"]),
            }
            (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            os.replace(checkpoint, directory / "R.bin")
            os.replace(staging / "metadata.json", directory / "metadata.json")
    return directory / "R.bin"


def write_ptq_metadata(result_dir, command):
    """Record each target experiment without adding target W to the cache key."""
    rotation_path = option(command, "--optimized_rotation_path")
    training = None
    checkpoint_digest = None
    if rotation_path is not None:
        cached = json.loads(Path(rotation_path).with_name("metadata.json").read_text())
        # Also accept completed caches created before rotation_config was added.
        training = rotation_config(cached["spec"]["command"])
        checkpoint_digest = cached["checkpoint_sha256"]
    metadata = {
        "model": option(command, "--input_model"),
        "target_w_bits": int(option(command, "--w_bits")),
        **{f"{name}_bits": int(option(command, f"--{name}_bits")) for name in "akv"},
        "quantizer": "rtn" if "--w_rtn" in command else "gptq",
        "rotation": "--rotate" in command,
        "had": "--r3" in command or "--r4" in command,
        "r3": "--r3" in command,
        "r4": "--r4" in command,
        "rotation_w_bits": training["rotation_w_bits"] if training else None,
        "max_steps": training["max_steps"] if training else None,
        "learning_rate": training["learning_rate"] if training else None,
        "seed": int(option(command, "--seed", "0")),
        "rotation_checkpoint": rotation_path,
        "rotation_checkpoint_sha256": checkpoint_digest,
        "command": command,
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    model_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(metadata["model"]).name)[:80]
    def switch(name):
        return "on" if metadata[name] else "off"
    bits = f"W{metadata['target_w_bits']}" + "".join(
        f"{name.upper()}{metadata[f'{name}_bits']}" for name in "akv"
    )
    label = (f"{model_name}_{bits}_{metadata['quantizer']}_rot-{switch('rotation')}"
             f"_had-{switch('had')}_r3-{switch('r3')}_r4-{switch('r4')}")
    if training:
        label += f"_steps{metadata['max_steps']}_rotW{metadata['rotation_w_bits']}"
    label += f"_seed-{metadata['seed']}_{key}"
    directory = Path(result_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}.metadata.json"
    # This file describes the requested run, not proof of successful evaluation.
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return directory / f"{label}.log"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cache-root")
    mode.add_argument("--ptq-result-dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.ptq_result_dir:
        print(write_ptq_metadata(args.ptq_result_dir, command))
    else:
        print(ensure_rotation(args.cache_root, command, force=args.force))


if __name__ == "__main__":
    main()
