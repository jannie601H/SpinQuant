#!/usr/bin/env python3
"""Build an R1 activation comparison from scratch, without any existing result files.

Stages: original FP reference, global rotation training, layer-wise rotation
training, FP R1 comparison. Training uses W16/A4/K4/V4; capture uses no
quantization. No GPTQ or full PPL evaluation is run.
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_experiments import code_version, now, run_stage, RUN_TIMEZONE
from scripts.capture_r1_comparison import sha256


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--input-model', default='meta-llama/Llama-3.2-1B')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--learning-rate', type=float, default=15.)
    p.add_argument('--dry-run', action='store_true', help='Print all commands without writes or model loading')
    p.add_argument('--resume', action='store_true', help='Skip completed stages; restart an incomplete stage')
    args = p.parse_args(argv)
    if args.max_steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error('max-steps and learning-rate must be positive')
    if not 0 <= args.seed < 2**32:
        p.error('seed must be between 0 and 2**32 - 1')
    args.output_dir = args.output_dir.resolve()
    if Path(args.input_model).is_dir():
        args.input_model = str(Path(args.input_model).resolve())
    return args


def build_stages(args):
    out = args.output_dir
    python = sys.executable
    launcher = [python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1']
    common = [
        '--input_model', args.input_model, '--model_max_length', '2048',
        '--fp16', 'False', '--bf16', 'True', '--save_safetensors', 'False',
        '--seed', str(args.seed), '--w_bits', '16', '--a_bits', '4', '--k_bits', '4', '--v_bits', '4',
        '--w_clip', '--a_asym', '--k_asym', '--v_asym', '--k_groupsize', '64', '--v_groupsize', '64',
        '--report_to', 'none', '--log_on_each_node', 'False', '--per_device_train_batch_size', '1',
        '--logging_steps', '1', '--learning_rate', str(args.learning_rate), '--weight_decay', '0.',
        '--lr_scheduler_type', 'cosine', '--gradient_checkpointing', 'True',
        '--max_steps', str(args.max_steps), '--save_strategy', 'no', '--disable_tqdm', 'True',
    ]
    stages = [('reference', [python, str(ROOT / 'scripts/capture_fp_activations.py'),
                           '--input-model', args.input_model, '--output-dir', str(out / 'reference'),
                           '--seed', str(args.seed), '--sequence-length', '2048',
                           '--dtype', 'bfloat16', '--device', 'cuda'])]
    for mode, flag in [('global', '--no-layerwise'), ('layerwise', '--layerwise')]:
        stages.append((mode, launcher + [str(ROOT / 'optimize_rotation.py')] + common + [
            flag, '--output_rotation_path', str(out / mode), '--output_dir', str(out / mode / 'output'),
            '--logging_dir', str(out / mode / 'logs'),
        ]))
    stages.append(('comparison', [python, str(ROOT / 'scripts/capture_r1_comparison.py'),
                                  '--reference-dir', str(out / 'reference'),
                                  '--global-rotation', str(out / 'global/R.bin'),
                                  '--layerwise-rotation', str(out / 'layerwise/R.bin'),
                                  '--output-dir', str(out / 'comparison'), '--device', 'cuda']))
    return stages


def artifact_hashes(directory, name):
    if name in ('global', 'layerwise'):
        files = [directory / 'R.bin']
        if files[0].stat().st_size == 0:
            raise ValueError(f'Empty checkpoint: {files[0]}')
    else:
        manifest = directory / ('manifest.json' if name == 'reference' else 'comparison.json')
        data = json.loads(manifest.read_text())
        if data.get('status') != 'complete':
            raise ValueError(f'Incomplete {name} capture: {manifest}')
        files = sorted(path for path in directory.rglob('*') if path.is_file())
        if name == 'reference' and not (directory / 'tokens.pt').exists():
            raise ValueError('Reference tokens are missing')
        if name == 'comparison' and not (directory / 'distribution.csv').exists():
            raise ValueError('Comparison statistics are missing')
    return {str(path.relative_to(directory)): sha256(path) for path in files}


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main(argv=None):
    args = parse_args(argv)
    stages = build_stages(args)
    print(f'R1 comparison from scratch: {args.output_dir}', flush=True)
    print(f'Global + layer-wise: W16/A4/K4/V4, LR={args.learning_rate}, {args.max_steps} steps each; '
          'FP capture: quantization off; no PPL', flush=True)
    if args.dry_run:
        for name, command in stages:
            print(f'{name}: {shlex.join(command)}')
        return 0
    path = args.output_dir / 'pipeline.json'
    commands = dict(stages)
    if args.resume:
        state = json.loads(path.read_text())
        if state['commands'] != commands:
            raise ValueError('Resume requires the same model, settings, paths and Python interpreter')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        state = {'status': 'running', 'started_at': now(), 'commands': commands,
                 'code': code_version(), 'stages': {}, 'training_weight_bits': 16,
                 'training_activation_kv_bits': 4, 'capture_quantization': False}
    state['status'] = 'running'
    state.pop('error', None)
    state.pop('finished_at', None)
    write_json(path, state)
    current = None
    try:
        for index, (name, command) in enumerate(stages, 1):
            current = name
            directory = args.output_dir / name
            previous = state['stages'].get(name, {})
            if previous.get('status') == 'complete':
                if artifact_hashes(directory, name) != previous['artifacts']:
                    raise ValueError(f'{name} artifacts changed; refusing to reuse them')
                print(f'[{index}/4] {name}: already complete, skipping', flush=True)
                continue
            attempt = previous.get('attempt', 0) + 1
            # Keep partial data for diagnosis; never mix partial captures/checkpoints.
            if directory.exists():
                backup = directory.with_name(name + '.incomplete.' + datetime.now(RUN_TIMEZONE).strftime('%Y%m%d_%H%M%S_%f'))
                directory.rename(backup)
                print(f'Preserved partial {name}: {backup}', flush=True)
            log = args.output_dir / f'{name}.attempt{attempt}.log'
            state['stages'][name] = {'status': 'running', 'attempt': attempt, 'started_at': now(), 'log': str(log)}
            write_json(path, state)
            print(f'[{index}/4] {name}', flush=True)
            returncode = run_stage(command, log, name)
            state['stages'][name]['returncode'] = returncode
            if returncode:
                raise RuntimeError(f'{name} exited with code {returncode}; see {log}')
            hashes = artifact_hashes(directory, name)
            if name in ('global', 'layerwise'):
                write_json(directory / 'result.json', {
                    'status': 'success', 'conditions': {'model': args.input_model,
                        'layerwise_flag': name == 'layerwise', 'learning_rate': args.learning_rate,
                        'max_steps': args.max_steps, 'seed': args.seed, 'w_bits': 16, 'a_bits': 4, 'kv_bits': 4},
                    'commands': {'train': command}, 'evaluation': None,
                    'note': 'Rotation training only; no weight PTQ or PPL evaluation', 'code': state['code'],
                })
            state['stages'][name].update(status='complete', finished_at=now(), artifacts=hashes)
            write_json(path, state)
        state.update(status='complete', finished_at=now())
        write_json(path, state)
    except (Exception, KeyboardInterrupt) as error:
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        state['error'] = str(error) or 'Interrupted'
        if current is not None and state['stages'].get(current, {}).get('status') == 'running':
            state['stages'][current]['status'] = state['status']
        write_json(path, state)
        print(f"{state['status']}: {state['error']}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    print(f'Complete: {args.output_dir / "comparison"}', flush=True)
    return 0


if __name__ == '__main__':
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    sys.exit(main())
