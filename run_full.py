import argparse
from pathlib import Path
from src.pipeline import run_experiment_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/real", help="Folder with result_{corpus}_{scenario}.csv files or PSMA repo root")
    ap.add_argument("--bootstrap-rounds", type=int, default=500)
    ap.add_argument("--max-train", type=int, default=60000, help="Limit for faster pilot. Use 0 for no cap.")
    ap.add_argument("--max-test", type=int, default=30000, help="Limit for faster pilot. Use 0 for no cap.")
    args = ap.parse_args()
    run_experiment_matrix(
        data_root=Path(args.data_root),
        scenarios=["online", "offline"],
        corpora=["rockyou", "linkedin", "000webhost"],
        out_prefix="",
        bootstrap_rounds=args.bootstrap_rounds,
        max_train=None if args.max_train == 0 else args.max_train,
        max_test=None if args.max_test == 0 else args.max_test,
    )

if __name__ == "__main__":
    main()
