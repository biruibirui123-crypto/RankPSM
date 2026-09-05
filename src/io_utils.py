from pathlib import Path
import hashlib, json, re
import pandas as pd
import numpy as np

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def possible_paths(root, corpus, scenario):
    root=Path(root)
    names=[f'result_{corpus}_{scenario}.csv', f'{corpus}_{scenario}.csv']
    bases=[root, root/'src'/'analyze', root/'analyze', root/'data', root/'results']
    for b in bases:
        for n in names:
            yield b/n

def find_result_file(root, corpus, scenario):
    for p in possible_paths(root,corpus,scenario):
        if p.exists(): return p
    raise FileNotFoundError(f'Cannot find result file for corpus={corpus}, scenario={scenario} under {root}')

def read_table_auto(path):
    path=Path(path)
    # Try common separators. Keep password as string and do not treat empty string as NaN.
    for sep in ['\t', ',', ';']:
        try:
            df=pd.read_csv(path,sep=sep,dtype={'password':str},keep_default_na=False,encoding_errors='replace')
            if len(df.columns)>=3: return df
        except Exception:
            pass
    return pd.read_csv(path,dtype={'password':str},keep_default_na=False,encoding_errors='replace')

def normalize_columns(df):
    cols={c.lower().strip():c for c in df.columns}
    # Flexible column mapping.
    password_col = cols.get('password') or cols.get('pw') or cols.get('passwd')
    strength_col = cols.get('strength') or cols.get('guess_number') or cols.get('guessnumber') or cols.get('score') or cols.get('target')
    weight_col = cols.get('weight') or cols.get('count') or cols.get('frequency') or cols.get('freq')
    if password_col is None or strength_col is None:
        raise ValueError(f'Missing required columns. Need password and strength-like column; got {list(df.columns)}')
    out=pd.DataFrame()
    out['password']=df[password_col].astype(str)
    out['strength']=pd.to_numeric(df[strength_col],errors='coerce')
    out['weight']=pd.to_numeric(df[weight_col],errors='coerce') if weight_col else 1.0
    # Preserve zxcvbn column if present.
    zcol=None
    for key,orig in cols.items():
        if 'zxcvbn' in key and ('guess' in key or 'score' in key or 'strength' in key):
            zcol=orig; break
    if zcol:
        out['zxcvbn']=pd.to_numeric(df[zcol],errors='coerce')
    out=out.dropna(subset=['password','strength','weight']).copy()
    out=out[out['password'].astype(str).str.len()>0]
    out=out[out['weight']>0]
    return out.reset_index(drop=True)

def load_result_dataset(root, corpus, scenario):
    p=find_result_file(root,corpus,scenario)
    raw=read_table_auto(p)
    df=normalize_columns(raw)
    df['corpus']=corpus; df['scenario']=scenario; df['source_file']=str(p)
    return df

def write_manifest(datasets, out_path):
    rows=[]
    for name,df in datasets.items():
        sf=df['source_file'].iloc[0] if 'source_file' in df and len(df) else ''
        rec={'dataset':name,'n_rows':int(len(df)),'n_unique_passwords':int(df['password'].nunique()),'has_zxcvbn':bool('zxcvbn' in df.columns)}
        if sf and Path(sf).exists():
            rec.update({'source_file':sf,'sha256':sha256_file(sf),'bytes':Path(sf).stat().st_size})
        rows.append(rec)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    return rows
