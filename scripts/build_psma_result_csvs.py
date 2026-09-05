"""Build result_{corpus}_{scenario}.csv files from the downloaded PSMA public files.

The output CSVs contain only columns needed by RankPSM. The script does not print
raw passwords to terminal.
"""
from pathlib import Path
import argparse, pandas as pd, json, hashlib
CORPORA=["rockyou","linkedin","000webhost"]
SCENARIOS=["online","offline"]

def read_lines(path: Path):
    return path.read_text(encoding='utf-8', errors='replace').splitlines()

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/psma_public_raw')
    ap.add_argument('--out', default='data/real')
    args=ap.parse_args()
    raw=Path(args.raw); out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    records=[]
    for scenario in SCENARIOS:
        for corpus in CORPORA:
            stem=f"{corpus}.{scenario}"
            ds=raw/'datasets'/scenario/corpus
            pw=ds/f"0_{stem}.pw"
            strength=ds/f"1_{stem}.strength"
            weight=ds/f"2_{stem}.weight"
            zguess=raw/'crawl'/'01_zxcvbn'/f"0_{stem}.pw_guess_number_result.txt"
            zscore=raw/'crawl'/'01_zxcvbn'/f"0_{stem}.pw_score_result.txt"
            passwords=read_lines(pw)
            strengths=pd.to_numeric(pd.Series(read_lines(strength)), errors='coerce')
            weights=pd.to_numeric(pd.Series(read_lines(weight)), errors='coerce')
            n=min(len(passwords), len(strengths), len(weights))
            df=pd.DataFrame({'password':passwords[:n], 'strength':strengths.iloc[:n], 'weight':weights.iloc[:n]})
            if zguess.exists():
                zg=pd.to_numeric(pd.Series(read_lines(zguess)), errors='coerce')
                df['zxcvbn_guess_number']=zg.iloc[:n].to_numpy()
            if zscore.exists():
                zs=pd.to_numeric(pd.Series(read_lines(zscore)), errors='coerce')
                df['zxcvbn_score']=zs.iloc[:n].to_numpy()
            # Do not print raw passwords; only aggregate metadata.
            df=df.dropna(subset=['password','strength','weight']).copy()
            df=df[df['password'].astype(str).str.len()>0]
            csv_path=out/f"result_{corpus}_{scenario}.csv"
            df.to_csv(csv_path, index=False)
            records.append({
                'corpus':corpus,'scenario':scenario,'rows':int(len(df)),
                'unique_passwords':int(df['password'].nunique()),
                'output_file':str(csv_path),'sha256':sha256(csv_path)
            })
            print(f"Built {csv_path}: rows={len(df)}, unique={df['password'].nunique()}")
    m=out/'psma_result_manifest.json'
    m.write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f"\nManifest: {m}")
if __name__=='__main__': main()
