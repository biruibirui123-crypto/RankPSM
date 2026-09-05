from pathlib import Path
import subprocess
import sys
from src.pipeline import run_experiment_matrix

ROOT = Path(__file__).resolve().parent
demo_dir = ROOT / "data" / "demo"

# Generate harmless synthetic demo data automatically if not present.
if not demo_dir.exists() or not list(demo_dir.glob("result_*_demo.csv")):
    subprocess.run([sys.executable, "scripts/make_demo_data.py"], check=True)

run_experiment_matrix(
    data_root=demo_dir,
    scenarios=["demo"],
    corpora=["demoA", "demoB", "demoC"],
    out_prefix="demo",
    bootstrap_rounds=100,
    max_train=50000,
    max_test=30000,
)
