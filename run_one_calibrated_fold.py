from pathlib import Path
import sys, time, argparse
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0,'.')
from src.io_utils import load_result_dataset
from src.metrics import evaluate, percentile_rank
from src.models import LengthModel, EntropyModel, LUDSModel, StructHGB, CharTFIDF, RankPSMLite

def signed_log1p_arr(a):
    a=np.asarray(a,dtype=float); return np.sign(a)*np.log1p(np.abs(a))

def grid_weights_three(yv, wv, rz, rs, rc, step=0.05):
    best=(-1e9,(1,0,0),None)
    vals=np.arange(0,1+1e-9,step)
    for wz in vals:
        for ws in vals:
            wc=round(1-wz-ws,10)
            if wc < -1e-9: continue
            pred=wz*rz+ws*rs+wc*rc
            met=evaluate(yv,pred,wv)
            obj=met['weighted_spearman']
            if obj>best[0]: best=(obj,(float(wz),float(ws),float(wc)),met)
    return best[1], best[2]

ap=argparse.ArgumentParser(); ap.add_argument('--scenario',required=True); ap.add_argument('--target',required=True); args=ap.parse_args()
ROOT=Path(__file__).resolve().parent; data_root=ROOT/'data'/'real'
corpora=['rockyou','linkedin','000webhost']; scenario=args.scenario; target=args.target; sources=[c for c in corpora if c!=target]
datasets={c: load_result_dataset(data_root,c,scenario) for c in corpora}
src=pd.concat([datasets[c] for c in sources],ignore_index=True)
train,valid=train_test_split(src,test_size=0.2,random_state=2026,shuffle=True)
train=train.reset_index(drop=True); valid=valid.reset_index(drop=True); test=datasets[target].reset_index(drop=True)
Xtr=train.password.astype(str).tolist(); ytr=signed_log1p_arr(train.strength); wtr=train.weight.to_numpy(float)
Xv=valid.password.astype(str).tolist(); yv=signed_log1p_arr(valid.strength); wv=valid.weight.to_numpy(float)
Xt=test.password.astype(str).tolist(); y=signed_log1p_arr(test.strength); w=test.weight.to_numpy(float)
models={'Length':LengthModel(),'Entropy':EntropyModel(),'LUDS':LUDSModel(),'Struct-HGB':StructHGB(),'Char-TFIDF':CharTFIDF()}
rows=[]; runtime=[]; preds={}
print('running',scenario,target,'n',len(train),len(valid),len(test),flush=True)
for name,m in models.items():
    t=time.time(); m.fit(Xtr,ytr,wtr); tr=time.time()-t
    t=time.time(); pred=m.predict(Xt); inf=time.time()-t
    preds[name]=pred; rows.append({'scenario':scenario,'train':'+'.join(sources),'test':target,'model':name,'n_train':len(train),'n_valid':len(valid),'n_test':len(test),'alpha':np.nan,**evaluate(y,pred,w)})
    runtime.append({'scenario':scenario,'test':target,'model':name,'train_seconds':tr,'infer_seconds':inf})
    print(name, rows[-1]['weighted_spearman'],flush=True)
rp=RankPSMLite(); t=time.time(); rp.fit(Xtr,ytr,wtr,valid=(Xv,yv,wv)); tr=time.time()-t
t=time.time(); fused,unc,ps,pc=rp.predict_components(Xt); inf=time.time()-t
for name,pred in {'RankPSM-Lite':fused,'Ablation-StructuralRank':percentile_rank(ps),'Ablation-CharRank':percentile_rank(pc)}.items():
    rows.append({'scenario':scenario,'train':'+'.join(sources),'test':target,'model':name,'n_train':len(train),'n_valid':len(valid),'n_test':len(test),'alpha':rp.alpha_ if name=='RankPSM-Lite' else np.nan,**evaluate(y,pred,w)})
    print(name, rows[-1]['weighted_spearman'],flush=True)
runtime.append({'scenario':scenario,'test':target,'model':'RankPSM-Lite','train_seconds':tr,'infer_seconds':inf,'alpha':rp.alpha_})
if 'zxcvbn' in test.columns:
    zt=test['zxcvbn'].fillna(test['zxcvbn'].median()).to_numpy(float)
    rows.append({'scenario':scenario,'train':'+'.join(sources),'test':target,'model':'zxcvbn-guess','n_train':len(train),'n_valid':len(valid),'n_test':len(test),'alpha':np.nan,**evaluate(y,zt,w)})
    print('zxcvbn-guess', rows[-1]['weighted_spearman'],flush=True)
    ps_v=rp.struct.predict(Xv); pc_v=rp.char.predict(Xv); z_v=valid['zxcvbn'].fillna(valid['zxcvbn'].median()).to_numpy(float)
    weights, valid_met=grid_weights_three(yv,wv,percentile_rank(z_v),percentile_rank(ps_v),percentile_rank(pc_v),step=0.05)
    wz,ws,wc=weights
    cal=wz*percentile_rank(zt)+ws*percentile_rank(ps)+wc*percentile_rank(pc)
    rows.append({'scenario':scenario,'train':'+'.join(sources),'test':target,'model':'RankPSM-Calibrated','n_train':len(train),'n_valid':len(valid),'n_test':len(test),'alpha':np.nan,'w_zxcvbn':wz,'w_struct':ws,'w_char':wc,**evaluate(y,cal,w)})
    print('RankPSM-Calibrated', rows[-1]['weighted_spearman'], weights, flush=True)
    Path('results_real_calibrated').mkdir(exist_ok=True)
    pd.DataFrame([{'scenario':scenario,'train':'+'.join(sources),'test':target,'w_zxcvbn':wz,'w_struct':ws,'w_char':wc,'valid_weighted_spearman':valid_met['weighted_spearman'],'valid_wsor20':valid_met['wsor20']}]).to_csv(f'results_real_calibrated/calibration_{scenario}_{target}.csv', index=False)
Path('results_real_calibrated').mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(f'results_real_calibrated/metrics_{scenario}_{target}.csv', index=False)
pd.DataFrame(runtime).to_csv(f'results_real_calibrated/runtime_{scenario}_{target}.csv', index=False)
