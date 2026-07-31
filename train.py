from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from vhenet.data import (
    FragmentTokenizer,
    ViralHostDataset,
    host_mapping,
    make_train_validation_split,
    read_fasta,
    read_host_similarity,
    read_interactions,
)
from vhenet.model import VHEModel
from vhenet.runtime import save_head_checkpoint
from vhenet.utils import load_config, project_path, select_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vhe_net.yaml"


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
    }
    return {key: float(value) for key, value in result.items()}


def evaluate(
    model: VHEModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels: list[float] = []
    probabilities: list[float] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validation", leave=False):
            logits = model(
                batch["seq"].to(device),
                batch["host_id"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probabilities.extend(torch.sigmoid(logits).cpu().view(-1).tolist())
            labels.extend(batch["label"].view(-1).tolist())
    return metrics(np.asarray(labels), np.asarray(probabilities))


def normalized_sample_weights(
    sample_weights: torch.Tensor,
    *,
    reference_mean: float,
    clip: float,
) -> torch.Tensor:
    """Sanitize and normalize weights against the fixed training-set mean."""
    sanitized = torch.nan_to_num(
        sample_weights,
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp_min(0.0)
    return (sanitized / max(float(reference_mean), 1e-6)).clamp(max=float(clip))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the single VHE-Net model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override the configured batch size.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = select_device(config.get("device", "auto"))

    data_config = config["data"]
    interaction_path = project_path(PROJECT_ROOT, data_config["interaction_csv"])
    similarity_path = project_path(PROJECT_ROOT, data_config["similarity_csv"])
    fasta_path = project_path(PROJECT_ROOT, data_config["fasta_path"])
    interactions = read_interactions(interaction_path)
    sequences = read_fasta(fasta_path)
    host_to_id = host_mapping(interactions)
    host_similarity = read_host_similarity(similarity_path, host_to_id)
    train_frame, validation_frame = make_train_validation_split(
        interactions,
        train_fraction=float(data_config["train_fraction"]),
        negative_fraction=float(data_config["negative_fraction"]),
        seed=seed,
    )

    missing_sequences = sorted(set(interactions["Virus"]).difference(sequences))
    if missing_sequences:
        raise ValueError(f"Training FASTA is missing {len(missing_sequences)} viruses.")

    model_config = config["model"]
    pretrained_model = project_path(PROJECT_ROOT, model_config["pretrained_model"])
    if not pretrained_model.exists():
        raise FileNotFoundError(
            f"LucaVirus model not found at {pretrained_model}. See README.md."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(pretrained_model),
        trust_remote_code=True,
    )
    fragment_tokenizer = FragmentTokenizer(
        tokenizer,
        max_length=int(config["sequence"]["max_length"]),
        num_fragments=int(config["sequence"]["num_fragments"]),
    )
    train_dataset = ViralHostDataset(
        train_frame,
        sequences,
        host_to_id,
        fragment_tokenizer,
    )
    validation_dataset = ViralHostDataset(
        validation_frame,
        sequences,
        host_to_id,
        fragment_tokenizer,
    )

    training_config = config["training"]
    batch_size = args.batch_size or int(training_config["batch_size"])
    epochs = args.epochs or int(training_config["epochs"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = VHEModel(
        num_hosts=len(host_to_id),
        pretrained_model=str(pretrained_model),
        host_similarity=host_similarity,
        embed_dim=int(model_config["embed_dim"]),
        num_heads=int(model_config["num_heads"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    positive_count = float((train_frame["Label"] == 1).sum())
    negative_count = float((train_frame["Label"] == 0).sum())
    positive_weight = torch.tensor(
        [negative_count / max(positive_count, 1.0)],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=positive_weight,
    )

    output_checkpoint = project_path(PROJECT_ROOT, model_config["checkpoint"])
    output_history = project_path(PROJECT_ROOT, training_config["history_path"])
    output_history.parent.mkdir(parents=True, exist_ok=True)
    patience = int(training_config["early_stopping_patience"])
    weight_clip = float(training_config["sample_weight_clip"])
    training_weights = torch.as_tensor(
        train_frame["Weight"].to_numpy(dtype=np.float32), dtype=torch.float32
    )
    training_weight_mean = float(
        torch.nan_to_num(
            training_weights,
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        )
        .clamp_min(0.0)
        .mean()
        .clamp_min(1e-6)
        .item()
    )
    best_auc = float("-inf")
    stale_epochs = 0
    history: list[dict] = []

    print(
        f"Training rows: {len(train_frame):,}; validation rows: "
        f"{len(validation_frame):,}; device: {device}; "
        f"training-weight mean: {training_weight_mean:.6f}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            logits = model(
                batch["seq"].to(device),
                batch["host_id"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            labels = batch["label"].to(device).view(-1, 1)
            sample_weights = normalized_sample_weights(
                batch["weight"].to(device).view(-1, 1),
                reference_mean=training_weight_mean,
                clip=weight_clip,
            )

            loss = (criterion(logits, labels) * sample_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_metrics = evaluate(model, validation_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **validation_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2))

        if validation_metrics["roc_auc"] > best_auc:
            best_auc = validation_metrics["roc_auc"]
            stale_epochs = 0
            save_head_checkpoint(
                model,
                output_checkpoint,
                metadata={
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "seed": seed,
                    "batch_size": batch_size,
                    "positive_class_weight": float(positive_weight.item()),
                    "sample_weight_reference_mean": training_weight_mean,
                    "sample_weight_clip": weight_clip,
                },
            )
            print(f"Saved best checkpoint: {output_checkpoint}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

        output_history.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    output_history.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training history: {output_history}")


if __name__ == "__main__":
    main()
