# Data directory

Do not commit raw password datasets or converted password CSV files.

To reproduce the experiments, download the official Password-Strength-Meter-Accuracy benchmark separately and keep it locally. Then build local CSV files with:

```bash
python scripts/build_psma_result_csvs.py --raw <PSMA_ROOT>/src --out data/real
```

The repository intentionally excludes:

- `Password-Strength-Meter-Accuracy-master.zip`
- `data/psma_raw/`
- `data/real/`
- `prediction_audit_private_v3_1/`
- any `*.pw` and `*.withcount` files
