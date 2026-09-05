
from pathlib import Path
import argparse, numpy as np, pandas as pd
from src.metrics import weighted_spearman

def bootstrap_ci(y,p,w,B=1000,seed=2026):
    rng=np.random.default_rng(seed); n=len(y); vals=[]
    for _ in range(B):
        idx=rng.integers(0,n,n)
        vals.append(weighted_spearman(y[idx],p[idx],w[idx]))
    lo,hi=np.percentile(vals,[2.5,97.5])
    return float(lo), float(hi)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred-dir', default='prediction_audit_private_v3_1')
    ap.add_argument('--out', default='results_v3_1/bootstrap_and_paired_tests.csv')
    ap.add_argument('--B', type=int, default=1000)
    args=ap.parse_args()
    pred_dir=Path(args.pred_dir); rows=[]
    for fp in sorted(pred_dir.glob('predictions_*.csv')):
        df=pd.read_csv(fp)
        tag=fp.stem.replace('predictions_','')
        # parse scenario target by first underscore, target may include digits.
        scenario, target = tag.split('_',1)
        y=df['strength'].to_numpy(float); w=df['weight'].to_numpy(float)
        for model in [c for c in df.columns if c not in ['password','strength','weight']]:
            lo,hi=bootstrap_ci(y,df[model].to_numpy(float),w,B=args.B)
            rows.append({'scenario':scenario,'test':target,'model':model,'weighted_spearman_lo95':lo,'weighted_spearman_hi95':hi})
        if 'RankPSM-Calibrated' in df.columns and 'zxcvbn-guess' in df.columns:
            rng=np.random.default_rng(2026); n=len(df); deltas=[]
            for _ in range(args.B):
                idx=rng.integers(0,n,n)
                cal=weighted_spearman(y[idx],df['RankPSM-Calibrated'].to_numpy(float)[idx],w[idx])
                z=weighted_spearman(y[idx],df['zxcvbn-guess'].to_numpy(float)[idx],w[idx])
                deltas.append(cal-z)
            lo,hi=np.percentile(deltas,[2.5,97.5]); p_two=2*min(np.mean(np.array(deltas)<=0), np.mean(np.array(deltas)>=0))
            rows.append({'scenario':scenario,'test':target,'model':'Delta:RankPSM-Calibrated_minus_zxcvbn','weighted_spearman_lo95':float(lo),'weighted_spearman_hi95':float(hi),'bootstrap_two_sided_p':float(min(1,p_two))})
    out=Path(args.out); out.parent.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print('Wrote',out)

if __name__ == '__main__':
    main()
