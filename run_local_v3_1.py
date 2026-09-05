
from pathlib import Path
import argparse, zipfile, subprocess, sys, shutil

def run(cmd):
    print('\n$', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--psma-zip', required=True, help='Path to Password-Strength-Meter-Accuracy-master.zip')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--max-train', type=int, default=None, help='Optional debug cap; omit for full run')
    ap.add_argument('--max-test', type=int, default=None, help='Optional debug cap; omit for full run')
    args=ap.parse_args()
    root=Path.cwd(); raw=root/'data'/'psma_raw'; real=root/'data'/'real'
    raw.mkdir(parents=True, exist_ok=True); real.mkdir(parents=True, exist_ok=True)
    # Extract zip
    if any(raw.iterdir()):
        print('data/psma_raw already exists; using existing extracted files.')
    else:
        print('Extracting PSMA zip...')
        with zipfile.ZipFile(args.psma_zip,'r') as z: z.extractall(raw)
    # Build result CSVs from official structure
    run([sys.executable, 'scripts/build_psma_result_csvs.py', '--raw', str(raw), '--out', str(real)])
    run([sys.executable, 'scripts/verify_no_identity_columns.py', '--data-root', str(real)])
    cmd=[sys.executable, 'run_rankpsm_full_outputs_v3_1.py', '--data-root', str(real)]
    if args.max_train: cmd += ['--max-train', str(args.max_train)]
    if args.max_test: cmd += ['--max-test', str(args.max_test)]
    run(cmd)
    run([sys.executable, 'bootstrap_significance_v3_1.py', '--B', str(args.bootstrap)])
    print('\nDone. Send these folders back for manuscript v3.2:')
    print('  results_v3_1/')
    print('  paper_exports_v3_1/')
    print('  figures_v3_1/')
    print('  prediction_audit_private_v3_1/  (private only; do not publish)')

if __name__ == '__main__':
    main()
