from __future__ import annotations

import gc
from pathlib import Path
from typing import Mapping

import pandas as pd
import torch
from transformers import AutoTokenizer

from .data import (
    FragmentTokenizer,
    host_mapping,
    read_host_similarity,
    read_interactions,
)
from .model import VHEModel
from .utils import project_path, select_device


def _checkpoint_state(path: Path):
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict) or not state:
        raise TypeError(f"Checkpoint does not contain a state dictionary: {path}")
    if all(key.startswith("module.") for key in state):
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
    return payload, state


def checkpoint_contains_encoder(checkpoint_path: str | Path) -> bool:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train a model first or supply --checkpoint."
        )
    payload, state = _checkpoint_state(path)
    result = any(key.startswith("llm.") for key in state)
    del payload, state
    gc.collect()
    return result


def load_checkpoint(
    model: VHEModel,
    checkpoint_path: str | Path,
) -> str:
    """Load either the archived full checkpoint or a compact VHE-Net head."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train a model first or supply --checkpoint."
        )

    payload, state = _checkpoint_state(path)
    contains_encoder = any(key.startswith("llm.") for key in state)
    try:
        incompatible = model.load_state_dict(state, strict=False, assign=True)
    except TypeError:
        incompatible = model.load_state_dict(state, strict=False)

    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    if contains_encoder:
        if missing:
            raise RuntimeError(f"Full checkpoint is missing keys: {missing[:10]}")
        checkpoint_type = "full"
    else:
        non_encoder_missing = [key for key in missing if not key.startswith("llm.")]
        if non_encoder_missing:
            raise RuntimeError(
                f"Head checkpoint is missing prediction keys: {non_encoder_missing[:10]}"
            )
        checkpoint_type = "head-only"

    del payload, state
    gc.collect()
    return checkpoint_type


def save_head_checkpoint(
    model: VHEModel,
    path: str | Path,
    *,
    metadata: Mapping | None = None,
) -> Path:
    """Save only VHE-Net-specific parameters; frozen LucaVirus weights are omitted."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("llm.")
    }
    torch.save(
        {
            "format": "vhe-net-head-v1",
            "state_dict": state,
            "metadata": dict(metadata or {}),
        },
        output_path,
    )
    return output_path


class Predictor:
    def __init__(
        self,
        config: dict,
        *,
        project_root: str | Path,
        checkpoint: str | Path | None = None,
        pretrained_model: str | Path | None = None,
        device: str | None = None,
    ):
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.device = select_device(device or config.get("device", "auto"))

        interaction_path = project_path(
            self.project_root,
            config["data"]["interaction_csv"],
        )
        similarity_path = project_path(
            self.project_root,
            config["data"]["similarity_csv"],
        )
        interactions = read_interactions(interaction_path)
        self.host_to_id = host_mapping(interactions)
        self.host_names = [
            host for host, _ in sorted(self.host_to_id.items(), key=lambda item: item[1])
        ]
        host_similarity = read_host_similarity(similarity_path, self.host_to_id)

        model_location = pretrained_model or config["model"]["pretrained_model"]
        model_location = project_path(self.project_root, model_location)
        if not model_location.exists():
            raise FileNotFoundError(
                "LucaVirus model directory not found: "
                f"{model_location}. See README.md for setup."
            )
        self.pretrained_model = model_location
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_location),
            trust_remote_code=True,
        )

        sequence_config = config["sequence"]
        self.fragment_tokenizer = FragmentTokenizer(
            self.tokenizer,
            max_length=sequence_config["max_length"],
            num_fragments=sequence_config["num_fragments"],
        )

        model_config = config["model"]
        checkpoint_location = checkpoint or model_config["checkpoint"]
        checkpoint_location = project_path(self.project_root, checkpoint_location)
        contains_encoder = checkpoint_contains_encoder(checkpoint_location)

        if contains_encoder:
            with torch.device("meta"):
                self.model = VHEModel(
                    num_hosts=len(self.host_to_id),
                    pretrained_model=str(model_location),
                    host_similarity=host_similarity,
                    embed_dim=model_config["embed_dim"],
                    num_heads=model_config["num_heads"],
                    dropout=model_config["dropout"],
                    load_pretrained_encoder=False,
                    empty_init=True,
                )
        else:
            self.model = VHEModel(
                num_hosts=len(self.host_to_id),
                pretrained_model=str(model_location),
                host_similarity=host_similarity,
                embed_dim=model_config["embed_dim"],
                num_heads=model_config["num_heads"],
                dropout=model_config["dropout"],
            )

        self.checkpoint_type = load_checkpoint(self.model, checkpoint_location)
        self.checkpoint = checkpoint_location
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def predict(
        self,
        sequences: Mapping[str, str],
        *,
        host_batch_size: int = 128,
    ) -> pd.DataFrame:
        if host_batch_size < 1:
            raise ValueError("host_batch_size must be positive.")

        records: list[dict] = []
        with torch.inference_mode():
            for virus_id, sequence in sequences.items():
                input_ids, attention_mask = self.fragment_tokenizer(sequence)
                virus_features = self.model.encode_virus(
                    input_ids.unsqueeze(0).to(self.device),
                    attention_mask.unsqueeze(0).to(self.device),
                )

                virus_records: list[dict] = []
                for start in range(0, len(self.host_names), host_batch_size):
                    names = self.host_names[start : start + host_batch_size]
                    host_ids = torch.tensor(
                        [self.host_to_id[name] for name in names],
                        dtype=torch.long,
                        device=self.device,
                    )
                    probabilities = torch.sigmoid(
                        self.model.score_encoded(virus_features, host_ids)
                    ).view(-1)
                    for host, probability in zip(
                        names,
                        probabilities.detach().cpu().tolist(),
                    ):
                        virus_records.append(
                            {
                                "Virus_ID": virus_id,
                                "Host_Name": host,
                                "Probability": float(probability),
                            }
                        )

                virus_records.sort(
                    key=lambda item: item["Probability"],
                    reverse=True,
                )
                for rank, record in enumerate(virus_records, start=1):
                    record["Rank"] = rank
                records.extend(virus_records)

                del input_ids, attention_mask, virus_features
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return pd.DataFrame.from_records(
            records,
            columns=["Virus_ID", "Host_Name", "Probability", "Rank"],
        )
