
from pathlib import Path
import sys, time, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
sys.path.insert(0, '.')
from src.io_utils import load_result_dataset, write_manifest
from src.metrics import evaluate, percentile_rank, weighted_spearman
from src.models import LengthModel, EntropyModel, LUDSModel, StructHGB, CharTFIDF, RankPSMLite

def signed_log1p_arr(a):
    a = np.asarray(a, dtype=float)
    return np.sign(a) * np.log1p(np.abs(a))

def grid_weights_three(yv, wv, rz, rs, rc, step=0.05, gamma=0.0):
    best = (-1e9, (1,0,0), None)
    vals = np.arange(0, 1+1e-9, step)
    for wz in vals:
        for ws in vals:
            wc = round(1 - wz - ws, 10)
            if wc < -1e-9:
                continue
            pred = wz*rz + ws*rs + wc*rc
            met = evaluate(yv, pred, wv)
            obj = met['weighted_spearman'] - gamma * met['wsor20']
            if obj > best[0]:
                best = (obj, (float(wz), float(ws), float(wc)), met)
    return best[1], best[2]

def risk_coverage(y, p, w, u, out_csv, out_png, title):
    rows=[]
    for cov in np.linspace(0.5, 1.0, 11):
        k=max(2,int(len(u)*cov)); idx=np.argsort(u)[:k]
        rows.append({'coverage':round(float(cov),2),'weighted_spearman':weighted_spearman(y[idx],p[idx],w[idx]),'n':int(k)})
    pd.DataFrame(rows).to_csv(out_csv,index=False)
    plt.figure(figsize=(6.2,4.0))
    plt.plot([r['coverage'] for r in rows],[r['weighted_spearman'] for r in rows], marker='o')
    plt.xlabel('Coverage'); plt.ylabel('Weighted Spearman'); plt.title(title)
    plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out_png,dpi=300); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/real')
    ap.add_argument('--out-root', default='.')
    ap.add_argument('--max-train', type=int, default=None, help='Optional cap for debugging; omit for full run.')
    ap.add_argument('--max-test', type=int, default=None, help='Optional cap for debugging; omit for full run.')
    ap.add_argument('--gamma-wsor', type=float, default=0.0, help='Optional source validation penalty for W-SOR in fusion search.')
    args=ap.parse_args()
    root=Path(args.out_root)
    OUT=root/'results_v3_1'; EXP=root/'paper_exports_v3_1'; FIG=root/'figures_v3_1'; PRED=root/'prediction_audit_private_v3_1'; LOG=root/'logs_v3_1'
    for d in [OUT,EXP,FIG,PRED,LOG]:
        d.mkdir(exist_ok=True, parents=True)
    corpora=['rockyou','linkedin','000webhost']; scenarios=['online','offline']
    rows=[]; runtime=[]; manifest={}; calibration_rows=[]
    for scenario in scenarios:
        datasets={c: load_result_dataset(args.data_root, c, scenario) for c in corpora}
        manifest.update({f'{scenario}:{c}': df for c,df in datasets.items()})
        for target in corpora:
            sources=[c for c in corpora if c != target]
            src=pd.concat([datasets[c] for c in sources], ignore_index=True)
            train,valid=train_test_split(src,test_size=0.2,random_state=2026,shuffle=True)
            if args.max_train: train=train.sample(min(args.max_train,len(train)), random_state=2026)
            test=datasets[target].reset_index(drop=True)
            if args.max_test: test=test.sample(min(args.max_test,len(test)), random_state=2026).reset_index(drop=True)
            train=train.reset_index(drop=True); valid=valid.reset_index(drop=True)
            Xtr=train.password.astype(str).tolist(); ytr=signed_log1p_arr(train.strength); wtr=train.weight.to_numpy(float)
            Xv=valid.password.astype(str).tolist(); yv=signed_log1p_arr(valid.strength); wv=valid.weight.to_numpy(float)
            Xt=test.password.astype(str).tolist(); y=signed_log1p_arr(test.strength); w=test.weight.to_numpy(float)
            print(f'FOLD {scenario}/{target}: train={len(train)} valid={len(valid)} test={len(test)}', flush=True)
            models={'Length':LengthModel(),'Entropy':EntropyModel(),'LUDS':LUDSModel(),'Struct-HGB':StructHGB(),'Char-TFIDF':CharTFIDF()}
            preds={}
            for name,m in models.items():
                t=time.time(); m.fit(Xtr,ytr,wtr); tr=time.time()-t
                t=time.time(); pred=m.predict(Xt); inf=time.time()-t
                preds[name]=pred
                rows.append({'scenario':scenario,'train':'+'.join(sources),'test':target,'model':name,'n_train':len(train),'n_valid':len(valid),'n_test':len(test),'alpha':np.nan,**evaluate(y,pred,w)})
                runtime.append({'scenario':scenario,'test':target,'model':name,'train_seconds':tr,'infer_seconds':inf})
                print(' ', name, rows[-1]['weighted_spearman'], flush=True)
            rp=RankPSMLite(); t=time.time(); rp.fit(Xtr,ytr,wtr,valid=(Xv,yv,wv)); tr=time.time()-t
            t=time.time(); fused,unc,ps,pc=rp.predict_components(Xt); inf=time.time()-t
            for name,pred in {'RankPSM-Lite':fused,'Ablation-StructuralRank':percentile_rank(ps),'Ablation-CharRank':percentile_rank(pc)}.items():
                preds[name]=pred
                rows.append({'scenario':scenario,'train':'+'.join(sources),'test':target,'model':name,'n_train':len(train),'n_valid':len(valid),'n_test':len(test),'alpha':rp.alpha_ if name=='RankPSM-Lite' else np.nan,**evaluate(y,pred,w)})
                print(' ', name, rows[-1]['weighted_spearman'], flush=True)
            runtime.append({'scenario':scenario,'test':target,'model':'RankPSM-Lite','train_seconds':tr,'infer_seconds':inf,'alpha':rp.alpha_})
            if 'zxcvbn' in test.columns and test['zxcvbn'].notna().any():
                zt=test['zxcvbn'].fillna(test['zxcvbn'].median()).to_numpy(float)
                preds['zxcvbn-guess']=zt
                rows.append({'scenario':scenario,'train':'+'.join(sources),'test':target,'model':'zxcvbn-guess','n_train':len(train),'n_valid':len(valid),'n_test':len(test),'alpha':np.nan,**evaluate(y,zt,w)})
                runtime.append({'scenario':scenario,'test':target,'model':'zxcvbn-guess','train_seconds':0.0,'infer_seconds':0.0})
                ps_v=rp.struct.predict(Xv); pc_v=rp.char.predict(Xv); z_v=valid['zxcvbn'].fillna(valid['zxcvbn'].median()).to_numpy(float)
                weights, valid_met=grid_weights_three(yv,wv,percentile_rank(z_v),percentile_rank(ps_v),percentile_rank(pc_v),step=0.05,gamma=args.gamma_wsor)
                wz,ws,wc=weights
                cal_pred=wz*percentile_rank(zt)+ws*percentile_rank(ps)+wc*percentile_rank(pc)
                cal_unc=np.average(np.vstack([
                    np.abs(percentile_rank(zt)-percentile_rank(ps)),
                    np.abs(percentile_rank(zt)-percentile_rank(pc)),
                    np.abs(percentile_rank(ps)-percentile_rank(pc))]),axis=0,weights=[max(wz,0.01),max(ws,0.01),max(wc,0.01)])
                preds['RankPSM-Calibrated']=cal_pred
                rows.append({'scenario':scenario,'train':'+'.join(sources),'test':target,'model':'RankPSM-Calibrated','n_train':len(train),'n_valid':len(valid),'n_test':len(test),'alpha':np.nan,'w_zxcvbn':wz,'w_struct':ws,'w_char':wc,**evaluate(y,cal_pred,w)})
                calibration_rows.append({'scenario':scenario,'train':'+'.join(sources),'test':target,'w_zxcvbn':wz,'w_struct':ws,'w_char':wc,'valid_weighted_spearman':valid_met['weighted_spearman'],'valid_wsor20':valid_met['wsor20']})
                risk_coverage(y,cal_pred,w,cal_unc,OUT/f'risk_coverage_{scenario}_{target}_calibrated.csv',FIG/f'risk_coverage_{scenario}_{target}_calibrated.png',f'Risk-Coverage {scenario}/{target}')
                print(' ', 'RankPSM-Calibrated', rows[-1]['weighted_spearman'], weights, flush=True)
            # private audit prediction file; do not upload publicly because it contains password strings.
            audit=pd.DataFrame({'password':test.password.astype(str),'strength':y,'weight':w})
            for k,v in preds.items(): audit[k]=v
            audit.to_csv(PRED/f'predictions_{scenario}_{target}.csv', index=False)
            pd.DataFrame(rows).to_csv(OUT/'all_metrics.csv',index=False)
            pd.DataFrame(runtime).to_csv(OUT/'runtime.csv',index=False)
            pd.DataFrame(calibration_rows).to_csv(OUT/'calibration_weights.csv',index=False)
    metrics=pd.DataFrame(rows)
    summary=metrics.groupby('model').agg(weighted_spearman_mean=('weighted_spearman','mean'),spearman_mean=('spearman','mean'),rank_mae_mean=('rank_mae','mean'),sor20_mean=('sor20','mean'),wsor20_mean=('wsor20','mean')).reset_index().sort_values('weighted_spearman_mean',ascending=False)
    summary.to_csv(OUT/'summary_table.csv',index=False)
    write_manifest(manifest, OUT/'data_manifest.json')
    metrics.pivot_table(index='model',columns=['scenario','test'],values='weighted_spearman').to_csv(EXP/'table_main_weighted_spearman.csv')
    metrics.pivot_table(index='model',columns=['scenario','test'],values='wsor20').to_csv(EXP/'table_security_overestimation.csv')
    metrics[metrics['model'].isin(['RankPSM-Calibrated','RankPSM-Lite','Ablation-StructuralRank','Ablation-CharRank','Struct-HGB','Char-TFIDF','zxcvbn-guess'])].groupby('model').agg(weighted_spearman_mean=('weighted_spearman','mean'),wsor20_mean=('wsor20','mean'),rank_mae_mean=('rank_mae','mean')).reset_index().to_csv(EXP/'table_ablation.csv',index=False)
    piv=metrics.pivot_table(index=['scenario','test'],columns='model',values='weighted_spearman')
    ax=piv.plot(kind='bar',figsize=(12,5.8)); ax.set_ylabel('Weighted Spearman'); ax.set_xlabel('Scenario / held-out corpus')
    plt.xticks(rotation=35,ha='right'); plt.tight_layout(); plt.savefig(FIG/'weighted_spearman.png',dpi=300); plt.close()
    piv2=metrics.pivot_table(index=['scenario','test'],columns='model',values='wsor20')
    ax=piv2.plot(kind='bar',figsize=(12,5.8)); ax.set_ylabel('W-SOR20 lower is better'); ax.set_xlabel('Scenario / held-out corpus')
    plt.xticks(rotation=35,ha='right'); plt.tight_layout(); plt.savefig(FIG/'wsor20.png',dpi=300); plt.close()
    print('\n=== Summary ===')
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
