import json
import os
from pathlib import Path
import subprocess
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
        fake.write_text('''#!/usr/bin/env python
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
        self.env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ['PATH'],
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

    def test_reuse_and_settings_separation(self):
        self.run_script('on', 'off', 'off')
        self.run_script('on', 'off', 'off', quantizer='rtn')
        self.assertEqual(len(self.calls('optimize_rotation.py')), 1)
        for command in self.calls('optimize_rotation.py') + self.calls('ptq.py'):
            self.assertIn('--no-r3', command)
            self.assertIn('--no-r4', command)
        self.run_script('on', 'off', 'off', env={'MAX_STEPS': '100'})
        self.run_script('on', 'on', 'off')
        self.run_script('on', 'off', 'on')
        self.run_script('on', 'off', 'off', model='other-org/model')
        self.run_script('on', 'off', 'off', env={'ROTATION_SEED': '1'})
        self.assertEqual(len(self.calls('optimize_rotation.py')), 6)
        self.assertEqual(len(list((self.root / 'results/rotation').glob('*/metadata.json'))), 6)
        self.run_script('on', 'off', 'off', env={'FORCE_ROTATION': '1'})
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
        self.assertEqual(len(self.calls('optimize_rotation.py')), 6)

    def test_off_and_invalid_arguments(self):
        self.run_script('off', 'off', 'off')
        self.assertEqual(len(self.calls('optimize_rotation.py')), 0)
        self.assertNotIn('--rotate', self.calls('ptq.py')[0])
        self.run_script('on', 'invalid', ok=False)
        self.run_script('on', quantizer='invalid', ok=False)
        self.assertEqual(len(self.calls('optimize_rotation.py')), 0)


if __name__ == '__main__':
    unittest.main()
