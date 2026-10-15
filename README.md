# Simulations of Silk Nanofibril Networks with Different Topologies (SNFNet-Topologies)

Numerical simulations of random, low-density core-arm (`ca_low`), and high-density core-arm (`ca_high`) networks. Includes irreversible fracture, reversible debonding and recovery, path metrics, and the original numerical reference data.

## Installation

Python 3.12 is recommended. From the repository directory:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux, then install:

```sh
python -m pip install .
```

## Reproduction

```sh
python -m corearm run --profile representative
python -m corearm run --profile manuscript
python -m corearm summarize --reference
python -m corearm verify results/manuscript
```

The representative profile runs three networks with seed 52. The manuscript profile runs 60 reversible simulations and six irreversible simulations. Each command uses a separate directory under `results/`; `--out PATH` selects another empty directory.

| Manuscript result | Simulations |
| --- | --- |
| Figure 7A | Network inputs and `network_metrics.csv` |
| Figure 7B–D | Irreversible boundary and box shear |
| Figure 7E–G | Reversible protocol and seed 42 |
| Figure 7H–K | Reversible seeds 42–61; mean and sample SD |
| Revised Figure 7L and Video S4 | Reversible seed 52 |

The 12.5% and 14.8% maximum-load debonding values belong to seed 42. Seed 52 gives 10.7% and 13.6% for `ca_low` and `ca_high`. These are individual-run values. The reference summary command separately calculates statistics from the original 20 runs per network.

Seed 52 was selected from the original runs with maximum-load debonding ordered `ca_high > ca_low > random`, minimizing the summed squared standardized deviations from the ensemble means of loading debonding, final debonding, and shape recovery across the three networks. `representative_selection.csv` records this selection.

## Data and outputs

`src/corearm/data/` contains the network inputs, protocol, source hashes, and original reference tables. Network ordering and random-number offsets are preserved. All protocol parameters are defined in `protocol.json`.

Each simulation saves `history.csv`, `summary.json`, and `state_trajectory.npz` containing coordinates and bond states at every step. Aggregate tables report response, debonding, recovery, replacement-path length, and closure demand. `peak_response_normalized = 1000 × peak_response_abs / total_edge_length`; 1000 is a dimensionless reporting scale and the response is in arbitrary units. Fractions are stored on a 0–1 scale.

`validation.json` checks numerical consistency. `reference_comparison.csv` reports differences from the original simulations separately; optimizer-dependent continuous results can vary across platforms. Original reference data are never replaced by rerun outputs.

## Tests

```sh
python -m unittest discover -s tests -v
```

