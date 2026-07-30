# VHE-Net

VHE-Net ranks candidate mammalian hosts for a viral nucleotide sequence. This
repository is the compact code-and-data release for the study's selected model
route.

Only one model is supported:

1. a frozen LucaVirus nucleotide encoder;
2. a 256-dimensional viral projection;
3. host ID and 451-host similarity features;
4. host-to-virus cross-attention;
5. a binary association classifier.

Historical model variants, ablation experiments, interpretability scripts, and
manuscript-generation files are intentionally excluded.

## Repository contents

```text
VHE-Net/
├── train.py                   # model training
├── predict.py                 # command-line prediction
├── configs/vhe_net.yaml       # the only supported configuration
├── data/
│   ├── interactions.csv       # 421,234 virus-host candidate pairs
│   ├── virus_sequences.fasta  # 934 viral sequences
│   └── host_similarity.csv    # 451 x 451 host matrix
└── vhenet/                    # reusable model and data code
```

The trained checkpoint and LucaVirus pretrained weights are not stored in this
Git repository.

The full archived checkpoint is approximately 3.5 GB. Running that
checkpoint is intended for a CUDA server or a machine with at least 24 GB RAM;
CPU inference is possible but slow. The compact checkpoint produced by
`train.py` does not duplicate the frozen encoder.

## Installation

Python 3.9 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the same LucaVirus model release used by the study in:

```text
pretrained/lucavirus/
```

The directory must contain its model weights, `config.json`, tokenizer
configuration, vocabulary, and the LucaVirus custom Python model/tokenizer
files. Alternatively, pass another local directory with
`--pretrained-model`.

## Train

```bash
python train.py
```

Training uses the fixed configuration in `configs/vhe_net.yaml`: seed 42, a
stratified 80/20 split after retaining all positives and 10% of candidate
negatives, 10 sequence fragments of 512 tokens, sample weighting, and
validation ROC AUC for checkpoint selection.

The saved file contains only the VHE-Net prediction head and host-similarity
buffer; the frozen LucaVirus weights are not duplicated.

For a short installation check:

```bash
python train.py --epochs 1 --batch-size 1
```

## Predict

```bash
python predict.py \
  --fasta examples/example.fasta \
  --checkpoint checkpoints/best_model.pt \
  --output outputs/predictions.csv
```

Each input virus is scored against all 451 hosts. The output contains the
virus identifier, host name, model probability, and within-virus rank. These
scores are association-ranking outputs, not estimates of clinical risk,
transmission, or causality.

The loader also accepts the archived full checkpoint when it is supplied
locally:

```bash
python predict.py \
  --fasta examples/example.fasta \
  --checkpoint /path/to/archived/best_model.pt \
  --pretrained-model /path/to/lucavirus
```

## Data notes

The interaction table contains 3,795 documented associations and 417,439
candidate unobserved pairs across 934 viruses and 451 hosts. Label 0 means
unobserved in the study data; it should not be interpreted as confirmed
biological non-association. See `data/README.md` for field definitions.

## Availability statement

The source code, processed interaction table, viral FASTA file, host-similarity
matrix, and fixed configuration needed to train and run VHE-Net are provided
in this repository. The trained checkpoint and third-party LucaVirus
pretrained weights are not committed because of their size. LucaVirus remains
subject to its original distribution terms.

## License

The VHE-Net source code is released under the MIT License. Third-party software
and pretrained models retain their original licenses.
