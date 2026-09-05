# RankPSM

Source-calibrated cross-corpus evaluation code for password strength meter analysis.

This repository accompanies the manuscript:

**RankPSM: Source-Calibrated Cross-Corpus Evaluation of Password Strength Meters**

## Scope

RankPSM evaluates password strength estimation under cross-corpus distribution shift. The implementation includes:

- structural password features
- character n-gram models
- source-calibrated rank fusion
- weighted Spearman evaluation
- severe overestimation analysis
- bootstrap confidence intervals and paired tests
- aggregate result export for manuscript tables and figures

## Data safety

This repository does **not** redistribute raw password datasets.

Raw benchmark files and converted password-level CSV files are intentionally excluded. Only code, aggregate metrics, tables, figures, and reproduction instructions are included.

Do not commit:

- `Password-Strength-Meter-Accuracy-master.zip`
- `data/psma_raw/`
- `data/real/`
- `prediction_audit_private_v3_1/`
- any file containing raw password strings, user names, emails, IP addresses, or account identifiers

## Installation

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Demo run

The demo uses harmless synthetic strings generated locally.

```bash
python run_demo.py
```

## Reproducing the benchmark experiments

Download the official Password-Strength-Meter-Accuracy benchmark separately, then build local CSV files:

```bash
python scripts/build_psma_result_csvs.py --raw <PSMA_ROOT>/src --out data/real
python scripts/verify_no_identity_columns.py --data-root data/real
```

Run the full experiment:

```bash
python run_rankpsm_full_outputs_v3_1.py --data-root data/real
```

Run bootstrap confidence intervals and paired tests:

```bash
python bootstrap_significance_v3_1.py --B 1000
```

## Included aggregate outputs

- `results/summary_table.csv`
- `results/all_metrics.csv`
- `results/bootstrap_and_paired_tests.csv`
- `results/calibration_weights.csv`
- `paper_exports/table_main_weighted_spearman.csv`
- `paper_exports/table_security_overestimation.csv`
- `paper_exports/table_ablation.csv`
- `figures/weighted_spearman.png`
- `figures/wsor20.png`
- `figures/risk_coverage_*.png`

## License

MIT License.
