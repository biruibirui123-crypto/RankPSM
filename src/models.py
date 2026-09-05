import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.features import frame
from src.metrics import percentile_rank, weighted_spearman

class LengthModel:
    def fit(self,X,y,w=None): return self
    def predict(self,X): return np.array([len(str(x)) for x in X],float)

class EntropyModel:
    def fit(self,X,y,w=None): return self
    def predict(self,X): return frame(X)['entropy_x_len'].to_numpy(float)

class LUDSModel:
    def fit(self,X,y,w=None): return self
    def predict(self,X):
        F=frame(X)
        return (F['length'] + 2*F['lower_count'].clip(0,1) + 2*F['upper_count'].clip(0,1) + 2*F['digit_count'].clip(0,1) + 2*F['special_count'].clip(0,1) - 3*F['keyboard_walk_ratio'] - 3*F['sequential_ratio']).to_numpy(float)

class StructHGB:
    def fit(self,X,y,w=None):
        F=frame(X); self.cols=list(F.columns)
        self.model=HistGradientBoostingRegressor(max_iter=260,learning_rate=0.055,max_leaf_nodes=31,l2_regularization=.35,random_state=42)
        try: self.model.fit(F,y,sample_weight=w)
        except TypeError: self.model.fit(F,y)
        return self
    def predict(self,X): return self.model.predict(frame(X)[self.cols])

class StructRF:
    def fit(self,X,y,w=None):
        F=frame(X); self.cols=list(F.columns)
        self.model=RandomForestRegressor(n_estimators=240,max_depth=18,min_samples_leaf=2,n_jobs=-1,random_state=42)
        self.model.fit(F,y,sample_weight=w)
        return self
    def predict(self,X): return self.model.predict(frame(X)[self.cols])

class CharTFIDF:
    def __init__(self):
        self.vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=120000,sublinear_tf=True,lowercase=False,dtype=np.float32)
        self.model=Ridge(alpha=2.0)
    def fit(self,X,y,w=None):
        M=self.vec.fit_transform([str(x) for x in X])
        try: self.model.fit(M,y,sample_weight=w)
        except TypeError: self.model.fit(M,y)
        return self
    def predict(self,X): return self.model.predict(self.vec.transform([str(x) for x in X]))

class RankPSMLite:
    def __init__(self, alpha_grid=None):
        self.alpha_grid = np.linspace(0,1,21) if alpha_grid is None else np.array(alpha_grid,float)
    def fit(self,X,y,w=None,valid=None):
        self.struct=StructHGB().fit(X,y,w)
        self.char=CharTFIDF().fit(X,y,w)
        if valid is None:
            self.alpha_=0.5
        else:
            Xv,yv,wv=valid
            ps=self.struct.predict(Xv); pc=self.char.predict(Xv)
            rs=percentile_rank(ps); rc=percentile_rank(pc)
            scores=[]
            for a in self.alpha_grid:
                pred=a*rc+(1-a)*rs
                scores.append(weighted_spearman(yv,pred,wv))
            self.alpha_=float(self.alpha_grid[int(np.nanargmax(scores))])
        return self
    def predict_components(self,X):
        ps=self.struct.predict(X); pc=self.char.predict(X)
        rs=percentile_rank(ps); rc=percentile_rank(pc)
        fused=self.alpha_*rc+(1-self.alpha_)*rs
        uncertainty=np.abs(rc-rs)
        return fused, uncertainty, ps, pc
    def predict(self,X): return self.predict_components(X)[0]
