import numpy as np
from scipy.stats import rankdata, spearmanr

def _clean(y,p,w=None):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    if w is None: w=np.ones_like(y,dtype=float)
    else: w=np.asarray(w,dtype=float)
    m=np.isfinite(y)&np.isfinite(p)&np.isfinite(w)&(w>0)
    return y[m],p[m],w[m]

def weighted_pearson(x,y,w):
    x,y,w=_clean(x,y,w)
    if len(x)<2 or w.sum()<=0: return np.nan
    w=w/w.sum(); mx=(w*x).sum(); my=(w*y).sum()
    cov=(w*(x-mx)*(y-my)).sum(); vx=(w*(x-mx)**2).sum(); vy=(w*(y-my)**2).sum()
    if vx<=0 or vy<=0: return np.nan
    return float(cov/np.sqrt(vx*vy))

def weighted_rank(values, weights):
    """Frequency-aware weighted midrank. Higher values receive higher ranks."""
    v=np.asarray(values,dtype=float); w=np.asarray(weights,dtype=float)
    order=np.argsort(v, kind='mergesort')
    ranks=np.empty_like(v,dtype=float)
    cum=0.0; i=0; n=len(v)
    while i<n:
        j=i+1
        while j<n and v[order[j]]==v[order[i]]: j+=1
        group=order[i:j]; wg=w[group].sum()
        ranks[group]=cum + 0.5*wg
        cum += wg; i=j
    return ranks

def weighted_spearman(y,p,w=None):
    y,p,w=_clean(y,p,w)
    if len(y)<2: return np.nan
    return weighted_pearson(weighted_rank(y,w), weighted_rank(p,w), w)

def percentile_rank(x):
    x=np.asarray(x,dtype=float)
    if len(x)<=1: return np.zeros_like(x)
    return (rankdata(x,method='average')-1)/(len(x)-1)

def severe_overestimation_rate(y,p,w=None,q=0.20):
    y,p,w=_clean(y,p,w)
    if len(y)<5: return {'sor20':np.nan,'wsor20':np.nan}
    yr=percentile_rank(y); pr=percentile_rank(p)
    weak=yr<=q; high=pr>=(1-q)
    if weak.sum()==0: return {'sor20':np.nan,'wsor20':np.nan}
    sor=float((weak&high).sum()/weak.sum())
    denom=w[weak].sum()
    wsor=float(w[weak&high].sum()/denom) if denom>0 else np.nan
    return {'sor20':sor,'wsor20':wsor}

def evaluate(y,p,w=None):
    y,p,w=_clean(y,p,w)
    rho=spearmanr(y,p,nan_policy='omit').statistic if len(y)>1 else np.nan
    wr=weighted_spearman(y,p,w)
    mae=float(np.average(np.abs(percentile_rank(y)-percentile_rank(p)),weights=w))
    sec=severe_overestimation_rate(y,p,w,q=0.20)
    return {'spearman':float(rho), 'weighted_spearman':float(wr), 'rank_mae':mae, **sec}

def bootstrap_ci(y,p,w,B=500,seed=2026,metric='weighted_spearman'):
    y,p,w=_clean(y,p,w)
    rng=np.random.default_rng(seed); vals=[]; n=len(y)
    if n<3: return (np.nan,np.nan)
    for _ in range(B):
        idx=rng.integers(0,n,n)
        vals.append(evaluate(y[idx],p[idx],w[idx])[metric])
    return float(np.nanpercentile(vals,2.5)), float(np.nanpercentile(vals,97.5))
