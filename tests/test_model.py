import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

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
