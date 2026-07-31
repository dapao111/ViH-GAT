from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import pandas as pd
import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


_REMOVED_SEQUENCE_CHARACTERS = re.compile(r"[-.*\s]")


def clean_sequence(sequence: str) -> str:
    cleaned = _REMOVED_SEQUENCE_CHARACTERS.sub("", sequence.upper())
    return cleaned or "N"


def _training_fasta_name(description: str) -> str:
    if "Accessions:" in description:
        return description.split("Accessions:", 1)[0].strip()
    return description.split("|", 1)[0].strip()


def read_fasta(path: str | Path) -> dict[str, str]:
    fasta_path = Path(path)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    sequences: dict[str, str] = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        name = _training_fasta_name(record.description)
        if not name:
            raise ValueError(f"Empty FASTA identifier in {fasta_path}")
        if name in sequences:
            raise ValueError(f"Duplicate FASTA identifier: {name}")
        sequences[name] = clean_sequence(str(record.seq))

    if not sequences:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    return sequences


def parse_fasta_text(text: str) -> dict[str, str]:
    sequences: dict[str, str] = {}
    identifier: str | None = None
    chunks: list[str] = []

    def store_record() -> None:
        if identifier is None:
            return
        if not chunks:
            raise ValueError(f"FASTA record has no sequence: {identifier}")
        sequences[identifier] = clean_sequence("".join(chunks))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            store_record()
            identifier = line[1:].split()[0]
            if not identifier:
                raise ValueError("FASTA identifier cannot be empty.")
            if identifier in sequences:
                raise ValueError(f"Duplicate FASTA identifier: {identifier}")
            chunks = []
        else:
            if identifier is None:
                raise ValueError("FASTA input must start with a '>' header.")
            chunks.append(line)

    store_record()
    if not sequences:
        raise ValueError("No FASTA records were provided.")
    return sequences


def read_interactions(path: str | Path) -> pd.DataFrame:
    interaction_path = Path(path)
    if not interaction_path.is_file():
        raise FileNotFoundError(f"Interaction table not found: {interaction_path}")

    frame = pd.read_csv(interaction_path)
    required = {"V-H", "Label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Interaction table is missing columns: {sorted(missing)}")

    names = frame["V-H"].astype(str).str.split("__", n=1, expand=True)
    if names.shape[1] != 2 or names.isna().any().any():
        raise ValueError("Every V-H value must have the form Virus__Host.")

    frame = frame.copy()
    frame["Virus"] = names[0]
    frame["Host"] = names[1]
    frame["Label"] = pd.to_numeric(frame["Label"], errors="raise").astype(float)
    if not set(frame["Label"].unique()).issubset({0.0, 1.0}):
        raise ValueError("Label must contain only 0 and 1.")
    if "Weight" not in frame:
        frame["Weight"] = 1.0
    frame["Weight"] = pd.to_numeric(frame["Weight"], errors="coerce").fillna(1.0)
    return frame


def host_mapping(frame: pd.DataFrame) -> dict[str, int]:
    hosts = sorted(frame["Host"].unique().tolist())
    return {host: index for index, host in enumerate(hosts)}


def read_host_similarity(
    path: str | Path,
    host_to_id: Mapping[str, int],
) -> torch.Tensor:
    similarity_path = Path(path)
    if not similarity_path.is_file():
        raise FileNotFoundError(f"Host similarity matrix not found: {similarity_path}")

    frame = pd.read_csv(similarity_path, index_col=0)
    ordered_hosts = [
        host for host, _ in sorted(host_to_id.items(), key=lambda item: item[1])
    ]
    missing = sorted(set(ordered_hosts).difference(frame.index))
    missing += sorted(set(ordered_hosts).difference(frame.columns))
    if missing:
        raise ValueError(f"Host similarity matrix is missing hosts: {sorted(set(missing))}")

    matrix = frame.loc[ordered_hosts, ordered_hosts].to_numpy(dtype="float32")
    if matrix.shape != (len(ordered_hosts), len(ordered_hosts)):
        raise ValueError(f"Unexpected host similarity shape: {matrix.shape}")
    return torch.from_numpy(matrix)


def make_train_validation_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    negative_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < negative_fraction <= 1:
        raise ValueError("negative_fraction must be between 0 and 1.")

    positives = frame[frame["Label"] == 1]
    negatives = frame[frame["Label"] == 0]
    if negative_fraction < 1:
        negatives = negatives.sample(frac=negative_fraction, random_state=seed)

    selected = pd.concat([positives, negatives], ignore_index=True)
    selected = selected.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_frame, validation_frame = train_test_split(
        selected,
        test_size=1 - train_fraction,
        random_state=seed,
        stratify=selected["Label"],
    )
    return train_frame.reset_index(drop=True), validation_frame.reset_index(drop=True)


class FragmentTokenizer:
    """Fixed 5-fragment, 512-token preprocessing for the selected model."""

    def __init__(self, tokenizer, *, max_length: int = 512, num_fragments: int = 5):
        if max_length < 4:
            raise ValueError("max_length must be at least 4.")
        if num_fragments < 1:
            raise ValueError("num_fragments must be positive.")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.num_fragments = int(num_fragments)

    def __call__(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = clean_sequence(sequence)
        window_size = self.max_length - 2
        step = max(1, int(window_size * 0.8))

        if len(sequence) <= window_size:
            fragments = [sequence]
        else:
            fragments = []
            for start in range(0, len(sequence), step):
                fragment = sequence[start : start + window_size]
                if len(fragment) >= 5:
                    fragments.append(fragment)
                if len(fragments) == self.num_fragments:
                    break

        encoded = self.tokenizer(
            fragments,
            seq_type="gene",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
            text_pair=None,
        )
        input_ids = encoded["input_ids"].long()
        attention_mask = encoded["attention_mask"].long()

        rows_to_pad = self.num_fragments - input_ids.size(0)
        if rows_to_pad > 0:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError("The tokenizer must define pad_token_id.")
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.full(
                        (rows_to_pad, self.max_length),
                        int(pad_token_id),
                        dtype=torch.long,
                    ),
                ]
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros((rows_to_pad, self.max_length), dtype=torch.long),
                ]
            )

        return (
            input_ids[: self.num_fragments],
            attention_mask[: self.num_fragments],
        )


class ViralHostDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        sequences: Mapping[str, str],
        host_to_id: Mapping[str, int],
        fragment_tokenizer: FragmentTokenizer,
    ):
        self.frame = frame.reset_index(drop=True)
        self.sequences = sequences
        self.host_to_id = host_to_id
        self.fragment_tokenizer = fragment_tokenizer
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        missing_sequences = sorted(set(self.frame["Virus"]).difference(sequences))
        if missing_sequences:
            preview = ", ".join(missing_sequences[:5])
            raise ValueError(f"Missing virus sequences ({len(missing_sequences)}): {preview}")
        missing_hosts = sorted(set(self.frame["Host"]).difference(host_to_id))
        if missing_hosts:
            raise ValueError(f"Missing host identifiers: {missing_hosts[:5]}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        virus = row["Virus"]
        if virus not in self._cache:
            self._cache[virus] = self.fragment_tokenizer(self.sequences[virus])
        input_ids, attention_mask = self._cache[virus]

        return {
            "seq": input_ids,
            "attention_mask": attention_mask,
            "host_id": torch.tensor(self.host_to_id[row["Host"]], dtype=torch.long),
            "label": torch.tensor(float(row["Label"]), dtype=torch.float32),
            "weight": torch.tensor(float(row["Weight"]), dtype=torch.float32),
        }
