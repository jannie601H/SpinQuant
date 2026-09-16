import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from eval_utils import main as eval_main, rotation_utils
from eval_utils.modeling_llama import LlamaForCausalLM
from train_utils import main as train_main, apply_r3_r4
from train_utils.modeling_llama_quant import LlamaForCausalLM as TrainLlama
from utils.process_args import parser_gen

ROOT = Path(__file__).resolve().parents[1]


class RotationSwitchTests(unittest.TestCase):
    def test_qk_math_and_quantization(self):
        # Test the actual wrappers with a CPU equivalent of the CUDA kernel.
        h = torch.tensor([[1.]])
        for _ in range(3):
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        config = LlamaConfig(hidden_size=16, num_attention_heads=2)
        torch.manual_seed(3)
        q = torch.randn(1, 2, 5, 8)
        k = torch.randn(1, 5, 2, 8).transpose(1, 2)  # RoPE may return noncontiguous K.
        for module in (rotation_utils, apply_r3_r4):
            for r3 in (False, True):
                for bits in (4, 16):
                    with self.subTest(module=module.__name__, r3=r3, bits=bits):
                        wrapper = module.QKRotationWrapper(
                            lambda: (q, k), config, r3=r3, k_bits=bits,
                            k_groupsize=8, k_sym=False, k_clip_ratio=1.0,
                        )
                        with patch.object(module.HadamardTransform, 'apply', side_effect=lambda x: x @ h) as had:
                            actual_q, actual_k = wrapper()
                        expected_q = q @ h / (8 ** 0.5) if r3 else q
                        expected_k = k @ h / (8 ** 0.5) if r3 else k
                        torch.testing.assert_close(actual_q, expected_q)
                        self.assertEqual(had.call_count, 2 if r3 else 0)
                        if bits == 16:
                            torch.testing.assert_close(actual_k, expected_k)
                        else:
                            self.assertFalse(torch.equal(actual_k, expected_k))
                            self.assertLess((actual_k - expected_k).abs().max().item(), 0.4)

    def test_pipeline_switches(self):
        config = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, vocab_size=32)
        for training in (False, True):
            for r3 in (False, True):
                for r4 in (False, True):
                    for bits in (4, 16):
                        with self.subTest(training=training, r3=r3, r4=r4, bits=bits):
                            with patch('sys.argv', ['test', '--rotate']):
                                args, _ = parser_gen()
                            args.r3, args.r4, args.k_bits = r3, r4, bits
                            args.k_groupsize = 64
                            model = (TrainLlama if training else LlamaForCausalLM)(config)
                            module = train_main if training else eval_main
                            rot = apply_r3_r4 if training else rotation_utils
                            with patch.object(rot, 'rotate_model') as offline, patch.object(module.utils, 'cleanup_memory'):
                                model = module.prepare_model(args, model) if training else module.ptq_model(args, model)
                            self.assertEqual(offline.call_count, int(r4) if training else 1)
                            layer = model.model.layers[0]
                            self.assertEqual(layer.mlp.down_proj.online_full_had, r4)
                            wrapper = getattr(layer.self_attn, 'apply_rotary_pos_emb_qk_rotation_wrapper', None)
                            self.assertEqual(wrapper is not None, r3 or bits < 16)
                            if wrapper is not None:
                                self.assertEqual(wrapper.r3, r3)
                                self.assertIs(layer.self_attn.forward.__func__.__globals__['apply_rotary_pos_emb'], wrapper)

    def test_r4_weight_gate_preserves_r1(self):
        # Execute the real weight-rotation function on CPU by mapping CUDA transfers.
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return original_to(tensor, *args, **kwargs)
        from types import SimpleNamespace
        for r4 in (False, True):
            linear = torch.nn.Linear(8, 8)
            weight = linear.weight.detach().clone()
            r1 = torch.eye(8).roll(1, 0)
            layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=linear))
            with patch.object(torch.Tensor, 'to', cpu_to), patch.object(rotation_utils, 'apply_exact_had_to_linear') as had:
                rotation_utils.rotate_mlp_output(layer, r1.double(), r4=r4)
            torch.testing.assert_close(linear.weight, r1.T @ weight)
            self.assertEqual(had.call_count, int(r4))

    def test_standalone_r4_preserves_logits(self):
        config = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, vocab_size=32)
        model = LlamaForCausalLM(config).eval()
        tokens = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            expected = model(tokens).logits
        with patch('sys.argv', ['test', '--r4', '--no-r3']):
            args, _ = parser_gen()
        h = torch.tensor([[1.]])
        for _ in range(8):
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h /= 16
        def rotate_weight(linear, **kwargs):
            linear.weight.data = linear.weight.data @ h
        with patch.object(eval_main.hadamard_utils, 'apply_exact_had_to_linear', side_effect=rotate_weight) as offline:
            model = eval_main.ptq_model(args, model)
        with patch.object(eval_main.hadamard_utils, 'matmul_hadU_cuda', side_effect=lambda x, *args: x @ h) as online:
            with torch.no_grad():
                actual = model(tokens).logits
        self.assertEqual(offline.call_count, 1)
        self.assertEqual(online.call_count, 1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


class ScriptCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fake = self.root / 'torchrun'
        fake.write_text(f'#!{sys.executable}\n' + '''
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(args) + '\\n')
if 'optimize_rotation.py' in args:
    target = pathlib.Path(args[args.index('--output_rotation_path') + 1])
    target.mkdir(parents=True, exist_ok=True)
    (target / 'R.bin').write_bytes(b'fake completed rotations')
    if os.environ.get('FAIL_TRAIN') == '1':
        sys.exit(1)
''')
        fake.chmod(0o755)
        python_runner = self.root / 'python-runner'
        python_runner.write_text(
            f'#!{sys.executable}\n'
            'import os, sys\n'
            'if sys.argv[1:3] == ["-m", "torch.distributed.run"]:\n'
            f'    os.execv({str(fake)!r}, [{str(fake)!r}, *sys.argv[3:]])\n'
            f'os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n'
        )
        python_runner.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ['PATH'],
                        PYTHON_BIN=str(python_runner),
                        RESULT_DIR=str(self.root / 'results'), CALLS=str(self.root / 'calls'),
                        MAX_STEPS='10', ROTATION_SEED='0', FORCE_ROTATION='0')

    def run_script(self, *switches, model='org/model', quantizer='gptq', env=None, ok=True,
                   w_bits='4', a_bits='8', k_bits='4', v_bits='4'):
        result = subprocess.run(
            ['bash', str(ROOT / 'scripts/run_ptq.sh'), model, w_bits, a_bits, k_bits, v_bits, quantizer, *switches],
            env=dict(self.env, **(env or {})), text=True, capture_output=True,
        )
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def calls(self, stage):
        path = self.root / 'calls'
        return [a for a in map(json.loads, path.read_text().splitlines()) if stage in a] if path.exists() else []

    @staticmethod
    def option(command, flag):
        # Duplicate precision arguments could silently override the intended bit.
        assert command.count(flag) == 1, command
        return command[command.index(flag) + 1]

    def test_gptq_protocol_and_metadata(self):
        for a, kv in [('8', '16'), ('8', '8'), ('4', '16'), ('4', '4')]:
            with self.subTest(a=a, kv=kv):
                self.run_script('on', a_bits=a, k_bits=kv, v_bits=kv)
                optimization = self.calls('optimize_rotation.py')[-1]
                ptq = self.calls('ptq.py')[-1]
                self.assertEqual(self.option(optimization, '--w_bits'), '16')
                self.assertEqual(self.option(ptq, '--w_bits'), '4')
                self.assertEqual(self.option(optimization, '--max_steps'), '10')
                for command in (optimization, ptq):
                    self.assertEqual(self.option(command, '--a_bits'), a)
                    self.assertEqual(self.option(command, '--k_bits'), kv)
                    self.assertEqual(self.option(command, '--v_bits'), kv)
                    self.assertIn('--r3' if int(kv) < 16 else '--no-r3', command)
                    self.assertIn('--r4', command)
                    self.assertNotIn('--w_rtn', command)
                self.assertIn('--rotate', ptq)
                checkpoint = Path(self.option(ptq, '--optimized_rotation_path'))
                self.assertTrue(checkpoint.is_file())
                self.assertIn(f'_W16A{a}K{kv}V{kv}_steps10_', checkpoint.parent.name)
                cached = json.loads(checkpoint.with_name('metadata.json').read_text())
                self.assertEqual(cached['rotation_config']['rotation_w_bits'], 16)
                self.assertNotIn('target_w_bits', cached['spec'])
                r3_label = 'on' if int(kv) < 16 else 'off'
                matches = list((self.root / 'results').glob(f'*_W4A{a}K{kv}V{kv}_gptq_rot-on_had-on_r3-{r3_label}_r4-on_steps10_*.metadata.json'))
                self.assertEqual(len(matches), 1)
                metadata = json.loads(matches[0].read_text())
                for field, expected in dict(rotation_w_bits=16, target_w_bits=4,
                                            a_bits=int(a), k_bits=int(kv), v_bits=int(kv),
                                            max_steps=10, quantizer='gptq', rotation=True,
                                            had=True, r3=int(kv) < 16, r4=True, seed=0).items():
                    self.assertEqual(metadata[field], expected)
                log = matches[0].with_name(matches[0].name.replace('.metadata.json', '.log')).read_text()
                self.assertIn('target_W=4', log)
                self.assertIn('rotation_w_bits=16 max_steps=10', log)

    def test_gptq_target_weights_share_cache_but_not_results(self):
        self.run_script('on', w_bits='4')
        self.run_script('on', w_bits='8')
        self.assertEqual(len(self.calls('optimize_rotation.py')), 1)
        first, second = self.calls('ptq.py')
        self.assertEqual(self.option(first, '--optimized_rotation_path'), self.option(second, '--optimized_rotation_path'))
        self.assertEqual(self.option(first, '--w_bits'), '4')
        self.assertEqual(self.option(second, '--w_bits'), '8')
        paths = list((self.root / 'results').glob('*.metadata.json'))
        self.assertEqual({json.loads(p.read_text())['target_w_bits'] for p in paths}, {4, 8})

    def test_respin_pipeline_and_separate_cache(self):
        self.run_script('on')
        self.run_script('on', env={'RESPIN': '1'})
        self.run_script('on', env={'RESPIN': '1'})
        self.assertEqual(len(self.calls('optimize_rotation.py')), 2)
        for stage in ('optimize_rotation.py', 'ptq.py'):
            self.assertNotIn('--respin', self.calls(stage)[0])
            self.assertIn('--respin', self.calls(stage)[-1])
        first, second, third = self.calls('ptq.py')
        self.assertNotEqual(self.option(first, '--optimized_rotation_path'), self.option(second, '--optimized_rotation_path'))
        self.assertEqual(self.option(second, '--optimized_rotation_path'), self.option(third, '--optimized_rotation_path'))
        path = Path(self.option(second, '--optimized_rotation_path'))
        self.assertTrue(json.loads(path.with_name('metadata.json').read_text())['rotation_config']['respin'])
        self.run_script('off', env={'RESPIN': '1'}, ok=False)

    def test_missing_python_reports_environment_fix(self):
        result = self.run_script('on', env={'PYTHON_BIN': str(self.root / 'missing-python')}, ok=False)
        self.assertIn('Set PYTHON_BIN', result.stderr)
        self.assertEqual(self.calls('optimize_rotation.py'), [])

    def test_python3_default_without_python_command(self):
        bin_dir = self.root / 'bin'
        bin_dir.mkdir()
        for command in ('bash', 'dirname', 'mkdir', 'tee'):
            (bin_dir / command).symlink_to(shutil.which(command))
        (bin_dir / 'python3').symlink_to(self.root / 'python-runner')
        self.run_script('on', env={'PATH': str(bin_dir), 'PYTHON_BIN': ''})
        self.assertEqual(len(self.calls('optimize_rotation.py')), 1)
        self.assertEqual(len(self.calls('ptq.py')), 1)

    def test_had_switch_controls_both_stages(self):
        for k in ('4', '8', '16'):
            for had in ('off', 'on'):
                with self.subTest(k=k, had=had):
                    self.run_script('on', had, k_bits=k)
                    r3 = had == 'on' and int(k) < 16
                    r4 = had == 'on'
                    for stage in ('optimize_rotation.py', 'ptq.py'):
                        command = self.calls(stage)[-1]
                        self.assertIn('--r3' if r3 else '--no-r3', command)
                        self.assertIn('--r4' if r4 else '--no-r4', command)
                        self.assertEqual(self.option(command, '--k_bits'), k)
                    checkpoint = Path(self.option(self.calls('ptq.py')[-1], '--optimized_rotation_path'))
                    config = json.loads(checkpoint.with_name('metadata.json').read_text())['rotation_config']
                    self.assertEqual((config['had'], config['r3'], config['r4']), (r4, r3, r4))
        self.assertEqual(len(self.calls('optimize_rotation.py')), 6)
        # Omitted HAD and explicit HAD=on resolve to the same cache.
        self.run_script('on', k_bits='16')
        self.assertEqual(len(self.calls('optimize_rotation.py')), 6)
        for path in (self.root / 'results').glob('*.metadata.json'):
            metadata = json.loads(path.read_text())
            self.assertEqual(metadata['r3'], metadata['had'] and metadata['k_bits'] < 16)
            self.assertEqual(metadata['r4'], metadata['had'])
            self.assertIn('_had-on_' if metadata['had'] else '_had-off_', path.name)

    def test_no_rotation_gptq_and_rtn(self):
        for quantizer, a, kv in [('gptq', '8', '16'), ('rtn', '4', '4')]:
            self.run_script('off', quantizer=quantizer, a_bits=a, k_bits=kv, v_bits=kv)
            self.assertEqual(len(self.calls('optimize_rotation.py')), 0)
            command = self.calls('ptq.py')[-1]
            self.assertEqual('--w_rtn' in command, quantizer == 'rtn')
            self.assertNotIn('--rotate', command)
            self.assertNotIn('--optimized_rotation_path', command)
            self.assertIn('--no-r3', command)
            self.assertIn('--no-r4', command)
            for flag, expected in [('w', '4'), ('a', a), ('k', kv), ('v', kv)]:
                self.assertEqual(self.option(command, f'--{flag}_bits'), expected)
        for path in (self.root / 'results').glob('*.metadata.json'):
            metadata = json.loads(path.read_text())
            self.assertFalse(metadata['rotation'])
            self.assertFalse(metadata['had'])
            self.assertFalse(metadata['r3'])
            self.assertFalse(metadata['r4'])
            self.assertIsNone(metadata['rotation_w_bits'])
            self.assertIsNone(metadata['max_steps'])

    def test_reuse_and_settings_separation(self):
        self.run_script('on', 'off')
        self.run_script('on', 'off', quantizer='rtn')
        self.assertEqual(len(self.calls('optimize_rotation.py')), 2)
        self.assertEqual(self.option(self.calls('optimize_rotation.py')[-1], '--w_bits'), '4')
        self.assertIn('--w_rtn', self.calls('ptq.py')[-1])
        for command in self.calls('optimize_rotation.py') + self.calls('ptq.py'):
            self.assertIn('--no-r3', command)
            self.assertIn('--no-r4', command)
        self.run_script('on', 'off', env={'MAX_STEPS': '100'})
        self.run_script('on', 'on')
        self.run_script('on', 'off', model='other-org/model')
        self.run_script('on', 'off', env={'ROTATION_SEED': '1'})
        self.assertEqual(len(self.calls('optimize_rotation.py')), 6)
        self.assertEqual(len(list((self.root / 'results/rotation').glob('*/metadata.json'))), 6)
        self.run_script('on', 'off', env={'FORCE_ROTATION': '1'})
        self.assertEqual(len(self.calls('optimize_rotation.py')), 7)

    def test_failed_training_and_corruption(self):
        self.run_script('on', env={'FAIL_TRAIN': '1'}, ok=False)
        self.assertEqual(len(self.calls('ptq.py')), 0)
        self.assertFalse(list((self.root / 'results/rotation').glob('*/R.bin')))
        self.run_script('on')
        self.run_script('on', env={'FORCE_ROTATION': '1', 'FAIL_TRAIN': '1'}, ok=False)
        self.run_script('on')  # Previous completed checkpoint survives failed force.
        self.assertEqual(len(self.calls('optimize_rotation.py')), 3)
        checkpoint = next((self.root / 'results/rotation').glob('*/R.bin'))
        checkpoint.write_bytes(b'corrupted')
        self.run_script('on')
        self.assertEqual(len(self.calls('optimize_rotation.py')), 4)

    def test_bits_and_local_model_changes(self):
        local_model = self.root / 'local model'
        local_model.mkdir()
        config = local_model / 'config.json'
        config.write_text('{}')
        self.run_script('on', model=str(local_model))
        self.run_script('on', model=str(local_model))
        self.assertEqual(len(self.calls('optimize_rotation.py')), 1)
        config.write_text('{"changed": true}')
        self.run_script('on', model=str(local_model))
        for setting in ('w_bits', 'a_bits', 'k_bits', 'v_bits'):
            self.run_script('on', model=str(local_model), **{setting: '16'})
        self.assertEqual(len(self.calls('optimize_rotation.py')), 5)

    def test_off_and_invalid_arguments(self):
        self.run_script('off', 'off')
        self.assertEqual(len(self.calls('optimize_rotation.py')), 0)
        self.assertNotIn('--rotate', self.calls('ptq.py')[0])
        self.run_script('on', 'invalid', ok=False)
        self.run_script('on', quantizer='invalid', ok=False)
        self.run_script('off', 'on', ok=False)
        self.run_script('on', 'off', 'off', ok=False)  # Obsolete 9-argument interface.
        self.assertEqual(len(self.calls('optimize_rotation.py')), 0)


if __name__ == '__main__':
    unittest.main()
