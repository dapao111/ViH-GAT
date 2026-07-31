import unittest

import pandas as pd
import torch

from vhenet.data import FragmentTokenizer, ViralHostDataset, parse_fasta_text


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, *, max_length, **kwargs):
        input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, text in enumerate(texts):
            values = [1] + [2 + (ord(char) % 4) for char in text] + [6]
            values = values[:max_length]
            input_ids[row, : len(values)] = torch.tensor(values)
            attention_mask[row, : len(values)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DataTests(unittest.TestCase):
    def test_default_fragment_count(self):
        self.assertEqual(FragmentTokenizer(FakeTokenizer()).num_fragments, 5)

    def test_fasta_text(self):
        records = parse_fasta_text(">virus_a\nAC GT\n>virus_b\nA-C.*T")
        self.assertEqual(records, {"virus_a": "ACGT", "virus_b": "ACT"})

    def test_fragment_shapes_and_dataset(self):
        fragment_tokenizer = FragmentTokenizer(
            FakeTokenizer(),
            max_length=8,
            num_fragments=3,
        )
        input_ids, mask = fragment_tokenizer("ACGT")
        self.assertEqual(tuple(input_ids.shape), (3, 8))
        self.assertEqual(tuple(mask.shape), (3, 8))
        self.assertEqual(int(mask[1:].sum()), 0)

        frame = pd.DataFrame(
            [{"Virus": "virus_a", "Host": "host_a", "Label": 1, "Weight": 2.5}]
        )
        dataset = ViralHostDataset(
            frame,
            {"virus_a": "ACGT"},
            {"host_a": 0},
            fragment_tokenizer,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["seq"].shape), (3, 8))
        self.assertEqual(sample["host_id"].item(), 0)
        self.assertAlmostEqual(sample["weight"].item(), 2.5)


if __name__ == "__main__":
    unittest.main()
