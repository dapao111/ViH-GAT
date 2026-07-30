from __future__ import annotations

import argparse
from pathlib import Path

from vhenet.data import read_fasta
from vhenet.runtime import Predictor
from vhenet.utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vhe_net.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate hosts for FASTA sequences.")
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "predictions.csv")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrained-model", type=Path)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--host-batch-size", type=int, default=128)
    args = parser.parse_args()

    config = load_config(args.config)
    predictor = Predictor(
        config,
        project_root=PROJECT_ROOT,
        checkpoint=args.checkpoint,
        pretrained_model=args.pretrained_model,
        device=args.device,
    )
    sequences = read_fasta(args.fasta)
    predictions = predictor.predict(
        sequences,
        host_batch_size=args.host_batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    for virus_id, frame in predictions.groupby("Virus_ID", sort=False):
        print(f"\n{virus_id}")
        print(frame.head(10).to_string(index=False))
    print(f"\nSaved {len(predictions):,} predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
