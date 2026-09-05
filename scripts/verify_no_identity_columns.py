from pathlib import Path
import argparse, pandas as pd, sys
BAD={'username','user','email','mail','ip','phone','name','account','uid','userid','hash','salt'}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root', default='data/real')
    args=ap.parse_args(); root=Path(args.data_root); ok=True
    for p in sorted(root.glob('*.csv')):
        try: cols=[c.lower().strip() for c in pd.read_csv(p,nrows=0).columns]
        except Exception as e: print(f'ERROR reading {p}: {e}'); ok=False; continue
        bad=[c for c in cols if c in BAD]
        if bad:
            print(f'WARNING {p} contains possible identity columns: {bad}'); ok=False
        else:
            print(f'OK {p.name}: no obvious identity columns')
    sys.exit(0 if ok else 1)
if __name__=='__main__': main()
