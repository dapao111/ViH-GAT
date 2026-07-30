# Data files

## `interactions.csv`

The table contains 421,234 unique candidate virus-host pairs covering 934
viruses and 451 mammalian hosts.

| Column | Meaning |
|---|---|
| `V-H` | Pair identifier in `Virus__Host` form |
| `Label` | 1 for a documented association; 0 for an unobserved candidate pair |
| `Virus_seq_count` | Virus sequence-record count |
| `Host_cite_count` | Host citation count |
| `Host_Dcite_count` | Host disease-related citation count |
| `Infected_host_overlap` | Host-overlap feature used during data construction |
| `Propensity_score` | Observation propensity score |
| `Weight` | Pair-level training weight |

An unobserved pair is not a confirmed biological negative.

## `virus_sequences.fasta`

One nucleotide sequence representation for each of the 934 viruses. FASTA
identifiers match the virus component of `V-H`.

## `host_similarity.csv`

A symmetric 451 x 451 host-similarity matrix. Row and column names match the
host component of `V-H`.

Run `shasum -a 256 -c data/SHA256SUMS` from the repository root to verify all
three published data files.
