#!/usr/bin/env python3
"""Run explicitly named rotation-training/PPL experiments in declaration order.

Edit EXPERIMENTS below to specify each experiment's complete conditions.
Use the Python environment containing SpinQuant's dependencies:

    python scripts/run_experiments.py --dry-run
    python scripts/run_experiments.py --experiments exp1 exp3 --dry-run

Remove --dry-run to execute. CUDA_VISIBLE_DEVICES selects the GPU; each stage
uses one torchrun worker. Without --experiments, every configured entry runs.
Results are written under results/experiments, relative to the repository.
Failures are recorded and execution continues; Ctrl-C/SIGTERM stops the run.
Exit codes: 0 = all succeeded (or dry run), 1 = failures, 130 = interrupted.

Full-model Trainer checkpoints and external metric reporters are disabled;
optimize_rotation.py still saves its final R.bin. Step losses stay in train.log.
Other training/evaluation defaults follow 10_optimize_rotation.sh/2_eval_ptq.sh.
The existing parser consumes --seed for rotation initialization and PTQ; the
Trainer's separate seed remains its existing default of 42.
"""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
MAX_STEPS = 3

# 실행할 실험을 아래에 직접 정의합니다. 각 항목은 한 번씩, 작성 순서대로 실행됩니다.
# 아래 exp1~exp3은 편집용 예시입니다. 항목을 복사해 exp4 등을 추가할 수 있습니다.
# model은 Hugging Face 모델 ID 또는 로컬 모델 디렉터리입니다.
# layerwise_flag에는 문자열이 아닌 Python의 True / False를 사용합니다.
EXPERIMENTS = {
    "exp1": {
        "model": "meta-llama/Llama-3.2-1B",
        "w_bits": 4,
        "a_bits": 4,
        "kv_bits": 4,
        "layerwise_flag": False,
        "learning_rate": 1.5,
    },
    "exp2": {
        "model": "meta-llama/Llama-3.2-1B",
        "w_bits": 4,
        "a_bits": 4,
        "kv_bits": 4,
        "layerwise_flag": True,
        "learning_rate": 1.5,
    },
    "exp3": {
        "model": "meta-llama/Llama-3.2-1B",
        "w_bits": 4,
        "a_bits": 4,
        "kv_bits": 4,
        "layerwise_flag": True,
        "learning_rate": 0.5,
    },
}

SUMMARY_FIELDS = [
    "experiment", "model", "w_bits", "a_bits", "kv_bits", "layerwise_flag",
    "learning_rate", "max_steps", "seed", "status", "final_ppl",
    "started_at", "finished_at", "duration_seconds", "result_dir", "error",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=list(EXPERIMENTS), metavar="NAME",
        help="Run only these names, in EXPERIMENTS order (default: all entries)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Rotation/PTQ seed (Trainer seed stays 42)")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--kv-groupsize", type=int, default=64,
        help="Existing default: 64. K requires the model head dimension or -1 (token-wise).",
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running or writing files")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    if args.kv_groupsize != -1 and args.kv_groupsize < 1:
        parser.error("--kv-groupsize must be positive or -1")
    args.results_dir = args.results_dir.resolve()
    return args


def conditions(args):
    required = {"model", "w_bits", "a_bits", "kv_bits", "layerwise_flag", "learning_rate"}
    selected = set(args.experiments) if args.experiments else set(EXPERIMENTS)
    if not selected:
        raise ValueError("EXPERIMENTS is empty; define at least one experiment")
    for name, definition in EXPERIMENTS.items():
        if name not in selected:
            continue
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", name):
            raise ValueError(f"Invalid experiment name {name!r}; use 1-32 letters, digits, '-' or '_'")
        if not isinstance(definition, dict) or set(definition) != required:
            raise ValueError(f"{name}: specify exactly these fields: {', '.join(sorted(required))}")
        config = dict(definition, name=name, max_steps=MAX_STEPS, seed=args.seed)
        if not isinstance(config["model"], str) or not config["model"].strip():
            raise ValueError(f"{name}: model must be a nonempty model ID or directory")
        for field in ("w_bits", "a_bits", "kv_bits"):
            if type(config[field]) is not int or not 2 <= config[field] <= 16:
                raise ValueError(f"{name}: {field} must be an integer between 2 and 16")
        if type(config["layerwise_flag"]) is not bool:
            raise ValueError(f"{name}: layerwise_flag must be True or False, not a string")
        rate = config["learning_rate"]
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"{name}: learning_rate must be a finite positive number")
        if Path(config["model"]).is_dir():
            config["model"] = str(Path(config["model"]).resolve())
        yield config


def experiment_name(config):
    model = re.sub(r"[^A-Za-z0-9._-]+", "-", config["model"]).strip(".-")[:100] or "model"
    return (
        f"{config['name']}_{model}_w{config['w_bits']}_a{config['a_bits']}_kv{config['kv_bits']}"
        f"_layerwise-{str(config['layerwise_flag']).lower()}_lr{config['learning_rate']}"
    )


def commands(config, directory, args):
    # Use this interpreter for both torchrun and its worker, so the active
    # environment supplies all dependencies. --standalone allocates a free port.
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=1"]
    common = [
        "--input_model", config["model"],
        "--model_max_length", "2048", "--fp16", "False", "--bf16", "True",
        "--save_safetensors", "False", "--seed", str(config["seed"]),
        "--w_bits", str(config["w_bits"]), "--a_bits", str(config["a_bits"]),
        "--k_bits", str(config["kv_bits"]), "--v_bits", str(config["kv_bits"]),
        "--layerwise" if config["layerwise_flag"] else "--no-layerwise",
        "--w_clip", "--a_asym", "--k_asym", "--v_asym",
        "--k_groupsize", str(args.kv_groupsize), "--v_groupsize", str(args.kv_groupsize),
        "--report_to", "none",
    ]
    train = launcher + [str(ROOT / "optimize_rotation.py")] + common + [
        "--output_rotation_path", str(directory),
        "--output_dir", str(directory / "output"),
        "--logging_dir", str(directory / "logs"),
        "--log_on_each_node", "False", "--per_device_train_batch_size", "1",
        "--logging_steps", "1", "--learning_rate", str(config["learning_rate"]),
        "--weight_decay", "0.", "--lr_scheduler_type", "cosine",
        "--gradient_checkpointing", "True", "--max_steps", str(MAX_STEPS),
        "--save_strategy", "no", "--disable_tqdm", "True",
    ]
    evaluate = launcher + [str(ROOT / "ptq.py")] + common + [
        "--output_dir", str(directory / "eval_output"),
        "--logging_dir", str(directory / "eval_logs"),
        "--do_train", "False", "--do_eval", "True",
        "--per_device_eval_batch_size", str(args.eval_batch_size),
        "--rotate", "--optimized_rotation_path", str(directory / "R.bin"),
    ]
    return {"train": train, "eval": evaluate}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_result(directory, result):
    temporary = directory / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "result.json")


def append_summary(results_dir, result):
    row = {key: result["conditions"].get(key, result.get(key)) for key in SUMMARY_FIELDS}
    with (results_dir / "summary.csv").open("a+", newline="", encoding="utf-8") as output:
        # Keep header and rows intact if separate runners share this directory.
        fcntl.flock(output, fcntl.LOCK_EX)
        output.seek(0, os.SEEK_END)
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
        if output.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        output.flush()


def code_version():
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "tracked_changes": git("status", "--short", "--untracked-files=no"),
    }


def run_stage(command, log_path, stage):
    print(f"  {stage}: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# started_at: {now()}\n# cwd: {ROOT}\n# command: {shlex.join(command)}\n\n")
        log.flush()
        environment = dict(os.environ, PYTHONUNBUFFERED="1")
        # PyTorch honors this variable when choosing the worker interpreter.
        environment["PYTHON_EXEC"] = sys.executable
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            return process.wait()
        except BaseException:
            # A terminal interrupt must not leave torchrun/GPU workers running.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise


def read_ppl(log_path):
    # ptq.py emits the full-precision value after evaluator() returns. Do not
    # use evaluator's rounded progress message or a training loss as the PPL.
    value = None
    with log_path.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            match = re.search(r"wiki2 ppl is:\s*(\S+)", line, re.IGNORECASE)
            if match:
                value = float(match.group(1))
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError("Evaluation did not report a finite positive final 'wiki2 ppl is:' value")
    return value


def run_experiment(config, directory, args, version):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "experiment": config["name"], "conditions": config,
        "status": "training", "final_ppl": None, "error": None,
        "started_at": now(), "finished_at": None, "duration_seconds": None,
        "result_dir": str(directory), "rotation_path": None,
        "return_codes": {"train": None, "eval": None},
        "commands": commands(config, directory, args),
        "evaluation": {
            "dataset": "Salesforce/wikitext", "subset": "wikitext-2-raw-v1",
            "split": "test", "sequence_length": 2048,
            "batch_size": args.eval_batch_size,
            "weight_quantization": "gptq" if config["w_bits"] < 16 else "none",
            "calibration_samples": 128,
        },
        "settings": {"trainer_seed": 42, "kv_groupsize": args.kv_groupsize},
        "code": version, "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_result(directory, result)
    stage = "train"
    try:
        code = run_stage(result["commands"][stage], directory / "train.log", stage)
        result["return_codes"][stage] = code
        if code != 0:
            raise RuntimeError(f"Training exited with code {code}; see train.log")
        rotation = directory / "R.bin"
        if not rotation.is_file() or rotation.stat().st_size == 0:
            raise RuntimeError("Training completed without a nonempty R.bin; see train.log")
        result["rotation_path"] = str(rotation)
        stage = "eval"
        result["status"] = "evaluating"
        write_result(directory, result)
        code = run_stage(result["commands"][stage], directory / "eval.log", stage)
        result["return_codes"][stage] = code
        if code != 0:
            raise RuntimeError(f"Evaluation exited with code {code}; see eval.log")
        result["final_ppl"] = read_ppl(directory / "eval.log")
        result["status"] = "success"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["error"] = f"Interrupted during {stage}"
    except (OSError, ValueError, RuntimeError) as error:
        result["status"] = f"{stage}_failed"
        result["error"] = str(error)
    finally:
        result["finished_at"] = now()
        result["duration_seconds"] = round(time.monotonic() - started, 3)
        write_result(directory, result)
        append_summary(args.results_dir, result)
    print(f"  {result['status']}: PPL={result['final_ppl']}" + (f"; {result['error']}" if result["error"] else ""), flush=True)
    return result["status"]


def main(argv=None):
    args = parse_args(argv)
    try:
        experiments = list(conditions(args))
    except ValueError as error:
        print(f"Invalid EXPERIMENTS configuration: {error}", file=sys.stderr)
        return 1
    print(f"{len(experiments)} experiments; max_steps={MAX_STEPS}; results={args.results_dir}", flush=True)
    if not args.dry_run and importlib.util.find_spec("torch") is None:
        print("PyTorch is unavailable. Run with the Python environment containing SpinQuant's dependencies.", file=sys.stderr)
        return 1
    version = code_version() if not args.dry_run else None
    failed = 0
    for index, config in enumerate(experiments, 1):
        suffix = "DRY_RUN" if args.dry_run else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
        directory = args.results_dir / f"{experiment_name(config)}_{suffix}"
        print(f"[{index}/{len(experiments)}] {directory.name}", flush=True)
        if args.dry_run:
            for stage, command in commands(config, directory, args).items():
                print(f"  {stage}: {shlex.join(command)}")
            continue
        status = run_experiment(config, directory, args, version)
        if status == "interrupted":
            return 130
        failed += status != "success"
    if not args.dry_run:
        print(f"Finished: {len(experiments) - failed} succeeded, {failed} failed. Summary: {args.results_dir / 'summary.csv'}", flush=True)
    return 1 if failed else 0


def interrupt(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
