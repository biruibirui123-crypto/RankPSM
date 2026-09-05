from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.io_utils import load_result_dataset, write_manifest
from src.metrics import evaluate, bootstrap_ci, weighted_spearman
from src.models import LengthModel, EntropyModel, LUDSModel, StructHGB, StructRF, CharTFIDF, RankPSMLite

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'; FIG=ROOT/'figures'; EXP=ROOT/'paper_exports'
for d in [RES,FIG,EXP]: d.mkdir(parents=True,exist_ok=True)

def cap(df,n):
    if n is None or len(df)<=n: return df.reset_index(drop=True)
    return df.sample(n=n,random_state=2026).reset_index(drop=True)

def split_source(df):
    tr,va=train_test_split(df,test_size=0.20,random_state=2026,shuffle=True)
    return tr.reset_index(drop=True), va.reset_index(drop=True)

def fit_predict_all(train, valid, test):
    Xtr=train.password.tolist(); ytr=train.strength.to_numpy(float); wtr=train.weight.to_numpy(float)
    Xv=valid.password.tolist(); yv=valid.strength.to_numpy(float); wv=valid.weight.to_numpy(float)
    Xt=test.password.tolist()
    models={
        'Length': LengthModel(),
        'Entropy': EntropyModel(),
        'LUDS': LUDSModel(),
        'Struct-HGB': StructHGB(),
        'Struct-RF': StructRF(),
        'Char-TFIDF': CharTFIDF(),
    }
    preds={}; runtimes={}
    for name,m in models.items():
        t=time.time(); m.fit(Xtr,ytr,wtr); train_time=time.time()-t
        t=time.time(); preds[name]=m.predict(Xt); infer_time=time.time()-t
        runtimes[name]={'train_seconds':train_time,'infer_seconds':infer_time}
    rp=RankPSMLite(); t=time.time(); rp.fit(Xtr,ytr,wtr,valid=(Xv,yv,wv)); train_time=time.time()-t
    t=time.time(); fused,unc,ps,pc=rp.predict_components(Xt); infer_time=time.time()-t
    preds['RankPSM-Lite']=fused; preds['Ablation-StructuralRank']=ps; preds['Ablation-CharRank']=pc
    runtimes['RankPSM-Lite']={'train_seconds':train_time,'infer_seconds':infer_time,'alpha':rp.alpha_}
    if 'zxcvbn' in test.columns and test['zxcvbn'].notna().any():
        preds['zxcvbn']=test['zxcvbn'].fillna(test['zxcvbn'].median()).to_numpy(float)
        runtimes['zxcvbn']={'train_seconds':0.0,'infer_seconds':0.0}
    return preds, runtimes, unc, getattr(rp,'alpha_',None)

def risk_coverage(y,p,w,u,tag):
    rows=[]
    for cov in np.linspace(.5,1.0,11):
        k=max(2,int(len(u)*cov)); idx=np.argsort(u)[:k]
        rows.append({'coverage':round(float(cov),2),'weighted_spearman':weighted_spearman(y[idx],p[idx],w[idx]),'n':int(k)})
    out=RES/f'risk_coverage_{tag}.csv'; pd.DataFrame(rows).to_csv(out,index=False)
    plt.figure(figsize=(6.4,4.2))
    plt.plot([r['coverage'] for r in rows],[r['weighted_spearman'] for r in rows],marker='o')
    plt.xlabel('Coverage'); plt.ylabel('Weighted Spearman'); plt.title(f'Risk-Coverage: {tag}')
    plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIG/f'risk_coverage_{tag}.png',dpi=300); plt.close()

def safe_tag(prefix, name):
    return (prefix+'_'+name if prefix else name).replace('__','_').strip('_')

def run_experiment_matrix(data_root, scenarios, corpora, out_prefix='', bootstrap_rounds=500, max_train=60000, max_test=30000):
    data_root=Path(data_root)
    all_metrics=[]; all_cis=[]; all_runtime=[]; all_preds=[]; manifest_datasets={}
    for scenario in scenarios:
        datasets={}
        for c in corpora:
            df=load_result_dataset(data_root,c,scenario)
            datasets[c]=df
            manifest_datasets[f'{scenario}:{c}']=df
        for target in corpora:
            sources=[c for c in corpora if c!=target]
            source=pd.concat([datasets[c] for c in sources],ignore_index=True)
            train,valid=split_source(source)
            train=cap(train,max_train); valid=cap(valid, max(5000, int((max_train or len(valid))*0.2)) if max_train else None); test=cap(datasets[target],max_test)
            preds,runtimes,unc,alpha=fit_predict_all(train,valid,test)
            y=test.strength.to_numpy(float); w=test.weight.to_numpy(float)
            tag=safe_tag(out_prefix,f'{scenario}_{target}')
            for name,pred in preds.items():
                m=evaluate(y,pred,w); lo,hi=bootstrap_ci(y,pred,w,B=bootstrap_rounds)
                all_metrics.append({'scenario':scenario,'train':'+'.join(sources),'test':target,'model':name,'n_train':len(train),'n_valid':len(valid),'n_test':len(test),'alpha':alpha if name=='RankPSM-Lite' else np.nan,**m})
                all_cis.append({'scenario':scenario,'test':target,'model':name,'weighted_spearman_lo95':lo,'weighted_spearman_hi95':hi})
                rt=dict(runtimes.get(name,{})); rt.update({'scenario':scenario,'test':target,'model':name})
                all_runtime.append(rt)
            if 'RankPSM-Lite' in preds:
                risk_coverage(y,preds['RankPSM-Lite'],w,unc,tag)
            # protected local predictions for audit only
            pr=pd.DataFrame({'scenario':scenario,'target':target,'password':test.password.astype(str),'strength':y,'weight':w,'rankpsm':preds['RankPSM-Lite'],'uncertainty':unc})
            all_preds.append(pr)
    metrics=pd.DataFrame(all_metrics); cis=pd.DataFrame(all_cis); runtime=pd.DataFrame(all_runtime)
    prefix = f'{out_prefix}_' if out_prefix else ''
    metrics.to_csv(RES/f'{prefix}all_metrics.csv',index=False)
    cis.to_csv(RES/f'{prefix}bootstrap_ci.csv',index=False)
    runtime.to_csv(RES/f'{prefix}runtime.csv',index=False)
    pd.concat(all_preds,ignore_index=True).to_csv(RES/f'{prefix}predictions_protected_research.csv',index=False)
    write_manifest(manifest_datasets, RES/f'{prefix}data_manifest.json')
    export_paper_tables(metrics,prefix)
    make_plots(metrics,prefix)
    print('\n=== Summary ===')
    summary=metrics.groupby('model').agg(weighted_spearman_mean=('weighted_spearman','mean'),spearman_mean=('spearman','mean'),rank_mae_mean=('rank_mae','mean'),sor20_mean=('sor20','mean'),wsor20_mean=('wsor20','mean')).reset_index().sort_values('weighted_spearman_mean',ascending=False)
    summary.to_csv(RES/f'{prefix}summary_table.csv',index=False)
    print(summary.to_string(index=False))
    print(f'\nResults folder: {RES}')
    print(f'Figures folder: {FIG}')
    print(f'Paper exports: {EXP}')

def export_paper_tables(metrics,prefix):
    # Main table: WSpearman by held-out corpus/scenario.
    main=metrics.pivot_table(index='model',columns=['scenario','test'],values='weighted_spearman')
    main.to_csv(EXP/f'{prefix}table_main_weighted_spearman.csv')
    sec=metrics.pivot_table(index='model',columns=['scenario','test'],values='wsor20')
    sec.to_csv(EXP/f'{prefix}table_security_overestimation.csv')
    # Ablation: include model names that are ablation-like and proposed.
    abl=metrics[metrics['model'].isin(['RankPSM-Lite','Ablation-StructuralRank','Ablation-CharRank','Struct-HGB','Char-TFIDF'])]
    abl.groupby('model').agg(weighted_spearman_mean=('weighted_spearman','mean'),wsor20_mean=('wsor20','mean'),rank_mae_mean=('rank_mae','mean')).reset_index().to_csv(EXP/f'{prefix}table_ablation.csv',index=False)
    # Generalization gap placeholder: this package computes cross-domain. In-domain can be added later.
    gd=metrics.groupby('model').agg(cross_domain_weighted_spearman=('weighted_spearman','mean'),cross_domain_wsor20=('wsor20','mean')).reset_index()
    gd['in_domain_weighted_spearman']='REQUIRED: run random-split in-domain experiment'
    gd['generalization_gap']='REQUIRED'
    gd.to_csv(EXP/f'{prefix}table_generalization_gap.csv',index=False)

def make_plots(metrics,prefix):
    # Weighted Spearman grouped bar
    piv=metrics.pivot_table(index=['scenario','test'],columns='model',values='weighted_spearman')
    ax=piv.plot(kind='bar',figsize=(11,5.5))
    ax.set_ylabel('Weighted Spearman'); ax.set_xlabel('Scenario / held-out corpus')
    plt.xticks(rotation=35,ha='right'); plt.tight_layout(); plt.savefig(FIG/f'{prefix}weighted_spearman.png',dpi=300); plt.close()
    piv2=metrics.pivot_table(index=['scenario','test'],columns='model',values='wsor20')
    ax=piv2.plot(kind='bar',figsize=(11,5.5))
    ax.set_ylabel('W-SOR20 lower is better'); ax.set_xlabel('Scenario / held-out corpus')
    plt.xticks(rotation=35,ha='right'); plt.tight_layout(); plt.savefig(FIG/f'{prefix}wsor20.png',dpi=300); plt.close()
