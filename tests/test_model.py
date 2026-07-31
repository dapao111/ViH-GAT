import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from train import normalized_sample_weights
from vhenet.model import VHEModel
from vhenet.runtime import load_checkpoint, save_head_checkpoint


class DummyConfig:
    hidden_size = 8


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)

    def forward(self, input_ids, **kwargs):
        return (self.embedding(input_ids),)


class ModelTests(unittest.TestCase):
    def test_sample_weights_use_fixed_reference_mean(self):
        weights = torch.tensor([[2.0], [4.0]])
        normalized = normalized_sample_weights(
            weights,
            reference_mean=3.0,
            clip=5.0,
        )
        torch.testing.assert_close(normalized, torch.tensor([[2.0 / 3.0], [4.0 / 3.0]]))

    @patch("vhenet.model.AutoModel.from_pretrained", return_value=DummyEncoder())
    @patch("vhenet.model.AutoConfig.from_pretrained", return_value=DummyConfig())
    def test_fragment_mask_excludes_padded_rows(self, _config, _model):
        model = VHEModel(
            num_hosts=2,
            pretrained_model="dummy",
            host_similarity=torch.eye(2),
            embed_dim=4,
            num_heads=2,
            dropout=0.0,
        )
        model.eval()
        features = torch.randn(1, 3, 4)
        fragment_mask = torch.tensor([[True, True, False]])

        with torch.no_grad():
            reference = model.score_encoded(
                features,
                torch.tensor([0]),
                fragment_mask,
            )
            altered = features.clone()
            altered[:, 2, :] = 1_000.0
            masked = model.score_encoded(
                altered,
                torch.tensor([0]),
                fragment_mask,
            )

        torch.testing.assert_close(reference, masked)

    @patch("vhenet.model.AutoModel.from_pretrained", return_value=DummyEncoder())
    @patch("vhenet.model.AutoConfig.from_pretrained", return_value=DummyConfig())
    def test_forward_and_head_checkpoint(self, _config, _model):
        model = VHEModel(
            num_hosts=2,
            pretrained_model="dummy",
            host_similarity=torch.eye(2),
            embed_dim=4,
            num_heads=2,
            dropout=0.0,
        )
        model.train()
        self.assertFalse(model.llm.training)

        input_ids = torch.randint(0, 31, (2, 3, 6))
        attention_mask = torch.ones_like(input_ids)
        logits = model(input_ids, torch.tensor([0, 1]), attention_mask)
        self.assertEqual(tuple(logits.shape), (2, 1))
        logits.mean().backward()
        self.assertIsNotNone(model.classifier[-1].weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.llm.parameters()))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "head.pt"
            save_head_checkpoint(model, checkpoint, metadata={"test": True})

            second_model = VHEModel(
                num_hosts=2,
                pretrained_model="dummy",
                host_similarity=torch.eye(2),
                embed_dim=4,
                num_heads=2,
                dropout=0.0,
            )
            checkpoint_type = load_checkpoint(second_model, checkpoint)
            self.assertEqual(checkpoint_type, "head-only")


if __name__ == "__main__":
    unittest.main()
