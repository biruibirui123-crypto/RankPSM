"""Download the public PSMA example benchmark files from GitHub.

This script downloads only the files published inside the official
RUB-SysSec/Password-Strength-Meter-Accuracy repository. It does not search for,
redistribute, or download additional credential leaks.

Usage:
    python scripts/download_psma_public_examples.py --out data/psma_public_raw

After download, run:
    python scripts/build_psma_result_csvs.py --raw data/psma_public_raw --out data/real
    python run_full.py --data-root data/real --bootstrap-rounds 500
"""
from pathlib import Path
import argparse, hashlib, json, urllib.request

BASE = "https://raw.githubusercontent.com/RUB-SysSec/Password-Strength-Meter-Accuracy/refs/heads/master/src"
CORPORA = ["rockyou", "linkedin", "000webhost"]
SCENARIOS = ["online", "offline"]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    req = urllib.request.Request(url, headers={"User-Agent":"RankPSM-research-script/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    path.write_bytes(data)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', default='data/psma_public_raw')
    args=ap.parse_args()
    out=Path(args.out)
    manifest=[]
    for scenario in SCENARIOS:
        for corpus in CORPORA:
            stem=f"{corpus}.{scenario}"
            ds_dir=out/'datasets'/scenario/corpus
            for prefix,ext in [(0,'pw'),(1,'strength'),(2,'weight'),(3,'withcount')]:
                rel=f"datasets/{scenario}/{corpus}/{prefix}_{stem}.{ext}"
                url=f"{BASE}/{rel}"
                path=out/rel
                download(url,path)
                manifest.append({"type":"dataset","corpus":corpus,"scenario":scenario,"file":str(path),"sha256":sha256(path),"bytes":path.stat().st_size})
            crawl_dir=out/'crawl'/'01_zxcvbn'
            for suffix in ['pw_guess_number_result.txt','pw_score_result.txt']:
                fname=f"0_{stem}.{suffix}"
                rel=f"crawl/01_zxcvbn/{fname}"
                url=f"{BASE}/{rel}"
                path=out/rel
                try:
                    download(url,path)
                    manifest.append({"type":"zxcvbn","corpus":corpus,"scenario":scenario,"file":str(path),"sha256":sha256(path),"bytes":path.stat().st_size})
                except Exception as e:
                    print(f"WARNING: could not download optional zxcvbn file {url}: {e}")
    mpath=out/'download_manifest.json'
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nDone. Manifest: {mpath}")

if __name__ == '__main__':
    main()
