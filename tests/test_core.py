from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

from tmt import TMTConfig, TMTModel, bytes_to_tensor, bytes_to_text, tensor_to_bytes
from tmt.config import load_config, parameter_count
from tmt.training import TMTTrainer, load_checkpoint, save_training_checkpoint
from tmt.training.checkpoint import restore_training_state
from tmt.training.streaming import IndependentByteBatcher


class ByteUtilityTests(unittest.TestCase):
    def test_utf8_round_trip(self) -> None:
        text = "hello \u2603"
        tensor = bytes_to_tensor(text)
        self.assertEqual(bytes_to_text(tensor), text)

    def test_invalid_utf8_decode_does_not_crash(self) -> None:
        self.assertIn("\ufffd", bytes_to_text(bytes([0xFF]), errors="replace"))
        self.assertEqual(tensor_to_bytes(torch.tensor([65, 66])), b"AB")


class StateTests(unittest.TestCase):
    def test_state_init_reset_detach_clone(self) -> None:
        model = TMTModel(TMTConfig(dim=8, num_layers=2))
        state = model.initial_state(batch_size=3)
        self.assertEqual(state.embedding_trace.shape, (3, 256, 8))
        state.layer_states[0][1].fill_(2.0)
        cloned = state.clone()
        self.assertTrue(torch.equal(cloned.layer_states[0], state.layer_states[0]))
        state.reset(mask=torch.tensor([False, True, False]))
        self.assertEqual(float(state.layer_states[0][1].sum()), 0.0)
        detached = cloned.detach()
        self.assertFalse(detached.layer_states[0].requires_grad)

    def test_state_isolation_between_streams(self) -> None:
        model = TMTModel(TMTConfig(dim=8, num_layers=1))
        state = model.initial_state(batch_size=2)
        output = model(torch.tensor([10, 20]), state)
        self.assertTrue(torch.all(output.state.embedding_trace[0, 10] == 1))
        self.assertTrue(torch.all(output.state.embedding_trace[1, 20] == 1))
        self.assertEqual(float(output.state.embedding_trace[0, 20].sum()), 0.0)
        self.assertEqual(float(output.state.embedding_trace[1, 10].sum()), 0.0)


class ModelTests(unittest.TestCase):
    def test_parameter_count_matches_original_formula(self) -> None:
        config = TMTConfig(dim=512, num_layers=16)
        model = TMTModel(config)
        self.assertEqual(model.parameter_count(), 4_481_793)
        self.assertEqual(model.parameter_count(), parameter_count(512, 16))

    def test_configs_load_with_expected_counts(self) -> None:
        expected = {
            "tmt_5m.yaml": 4_481_793,
            "tmt_25m.yaml": 24_713_473,
            "tmt_50m.yaml": 49_924_097,
            "tmt_100m.yaml": 100_635_393,
        }
        for filename, count in expected.items():
            with self.subTest(filename=filename):
                config = load_config(Path("configs") / filename)
                self.assertEqual(config.parameter_count_formula, count)

    def test_deterministic_forward_without_sampling(self) -> None:
        torch.manual_seed(123)
        model = TMTModel(TMTConfig(dim=16, num_layers=2))
        state = model.initial_state(batch_size=1)
        out_a = model(torch.tensor(42), state.clone())
        out_b = model(torch.tensor(42), state.clone())
        self.assertTrue(torch.allclose(out_a.logits, out_b.logits))
        self.assertTrue(torch.allclose(out_a.latent, out_b.latent))

    def test_training_step_computes_all_losses_and_detaches_state(self) -> None:
        model = TMTModel(TMTConfig(dim=16, num_layers=2))
        trainer = TMTTrainer(model)
        state = model.initial_state(batch_size=1)
        result = trainer.step(torch.tensor(65), state, next_byte=torch.tensor(66), end=torch.tensor(False), sample=True)
        self.assertGreater(result.metrics["loss_total"], 0.0)
        self.assertGreater(result.metrics["loss_byte"], 0.0)
        self.assertGreaterEqual(result.metrics["loss_variance"], 0.0)
        self.assertEqual(result.sampled_byte.shape, (1,))
        self.assertFalse(result.state.layer_states[0].requires_grad)

    def test_generation_smoke(self) -> None:
        model = TMTModel(TMTConfig(dim=16, num_layers=1))
        trainer = TMTTrainer(model)
        state = model.initial_state(batch_size=1)
        result = trainer.step(torch.tensor(10), state, update_weights=False, sample=True)
        self.assertIsNotNone(result.sampled_byte)
        self.assertGreaterEqual(int(result.sampled_byte.item()), 0)
        self.assertLessEqual(int(result.sampled_byte.item()), 255)


class CheckpointTests(unittest.TestCase):
    def test_training_checkpoint_round_trip(self) -> None:
        config = TMTConfig(dim=16, num_layers=2)
        model = TMTModel(config)
        trainer = TMTTrainer(model, config)
        state = model.initial_state(batch_size=1)
        state = trainer.step(torch.tensor(1), state, next_byte=torch.tensor(2), end=torch.tensor(False)).state

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.safetensors"
            torch.manual_seed(321)
            save_training_checkpoint(
                path,
                model,
                trainer.optimizer,
                state,
                config,
                step=7,
                run_id="test-run",
                progress={"bytes_processed": 1},
            )
            expected_random = torch.rand(4)
            checkpoint = load_checkpoint(path)

            restored_model = TMTModel(config)
            restored_trainer = TMTTrainer(restored_model, config)
            restored_state, step = restore_training_state(restored_model, restored_trainer.optimizer, checkpoint)
            actual_random = torch.rand(4)

            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()

        self.assertEqual(step, 7)
        self.assertEqual(checkpoint["run_id"], "test-run")
        self.assertEqual(checkpoint["progress"]["bytes_processed"], 1)
        self.assertEqual(metadata["format"], "tmt-safetensors")
        self.assertEqual(metadata["kind"], "training")
        self.assertTrue(torch.equal(expected_random, actual_random))
        self.assertIsNotNone(restored_state)
        for left, right in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.allclose(left, right))
        assert restored_state is not None
        self.assertTrue(torch.allclose(state.embedding_trace.cpu(), restored_state.embedding_trace))

    def test_checkpoint_requires_safetensors_extension(self) -> None:
        config = TMTConfig(dim=8, num_layers=1)
        model = TMTModel(config)
        trainer = TMTTrainer(model, config)
        state = model.initial_state(batch_size=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                save_training_checkpoint(
                    Path(tmpdir) / "checkpoint.pt",
                    model,
                    trainer.optimizer,
                    state,
                    config,
                    step=0,
                )


class StreamingResumeTests(unittest.TestCase):
    def test_batcher_resumes_at_exact_byte_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.txt"
            second = Path(tmpdir) / "second.txt"
            first.write_bytes(b"abcdef")
            second.write_bytes(b"uvwxyz")
            files = [first, second]

            batcher = IndependentByteBatcher(files, batch_size=1, repeat=True)
            next(batcher)
            next(batcher)
            saved_state = batcher.state_dict()
            expected = next(batcher)

            restored = IndependentByteBatcher(files, batch_size=1, repeat=True)
            restored.load_state_dict(saved_state)
            actual = next(restored)

        self.assertTrue(torch.equal(expected.current, actual.current))
        self.assertTrue(torch.equal(expected.next, actual.next))
        self.assertTrue(torch.equal(expected.end, actual.end))
        self.assertTrue(torch.equal(expected.new_document, actual.new_document))


class DeviceTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_execution_if_available(self) -> None:
        config = TMTConfig(dim=16, num_layers=1)
        model = TMTModel(config).cuda()
        trainer = TMTTrainer(model, config)
        state = model.initial_state(batch_size=1, device="cuda")
        result = trainer.step(torch.tensor(1, device="cuda"), state, next_byte=torch.tensor(2, device="cuda"))
        self.assertEqual(result.state.device.type, "cuda")

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 is not available",
    )
    def test_bf16_autocast_if_supported(self) -> None:
        config = TMTConfig(dim=16, num_layers=1)
        config.precision.autocast = True
        config.precision.dtype = "bf16"
        model = TMTModel(config).cuda()
        trainer = TMTTrainer(model, config)
        state = model.initial_state(batch_size=1, device="cuda")
        result = trainer.step(torch.tensor(1, device="cuda"), state, next_byte=torch.tensor(2, device="cuda"))
        self.assertFalse(math.isnan(result.metrics["loss_total"]))


if __name__ == "__main__":
    unittest.main()
