import os,sys,json,cv2,numpy as np,pandas as pd
from multiprocessing import Pool
from PIL import Image
sys.path.insert(0,os.path.expanduser('~/aic2026/code_vadim/src'))
from common import data_dir, load_split
D=data_dir()
def P(p):
    for b in (D, D/'stage1', D/'nanobanana', D/'test_stage1'):
        if (b/p).exists(): return b/p
    return D/p
def feats(p):
    try:
        im=Image.open(P(p)); w,h=im.size; qt=getattr(im,'quantization',None); q=np.array(qt[0][:64],float) if qt else np.zeros(64)
        sub=str(getattr(im,'layer',''))[:20]
        g=cv2.imread(str(P(p)),0).astype(np.float32)
        hp=g-cv2.GaussianBlur(g,(0,0),1.0); a=np.abs(hp)
        # шумовые статистики на плоских участках (низкий градиент) — там видно сенсорный шум vs перегенерация
        gx=cv2.Sobel(g,cv2.CV_32F,1,0); gy=cv2.Sobel(g,cv2.CV_32F,0,1); flat=(np.abs(gx)+np.abs(gy))<8
        fa=a[flat] if flat.sum()>1000 else a.ravel()
        return [w,h,w/h,float(max(w,h)==1024),float(q.mean()),float(q[0]),float(q[1]),float(np.median(a)),float(np.mean(a>2)),float(np.median(fa)),float(np.mean(fa>1)),float(np.mean(fa>3)),float(np.std(fa)),float(flat.mean())]
    except Exception as e:
        return None
if __name__=='__main__':
    sp=load_split('v1',None); s1=sp[sp.role!='train'].sample(3000,random_state=0)   # stage1 val-часть (train не нужен): позитивы+негативы всех доменов
    nb=pd.read_csv(D/'nanobanana/pico_nanobanana.csv').fillna('')
    nbo=nb.orgl_img_path.sample(1500,random_state=0).tolist(); nbe=nb.chng_img_path.sample(1500,random_state=1).tolist()
    test=pd.read_csv(D/'test_stage1/test.csv').img_path.tolist()
    items=[(p,0,d) for p,d in zip(s1.chng_img_path,s1.domain)]+[(p,1,'nb_orig') for p in nbo]+[(p,1,'nb_edit') for p in nbe]+[(p,-1,'test') for p in test]
    with Pool(32) as pool: F=pool.map(feats,[i[0] for i in items],chunksize=32)
    X=np.array([f for f in F if f]); y=np.array([i[1] for i,f in zip(items,F) if f]); dom=np.array([i[2] for i,f in zip(items,F) if f])
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    m=y>=0; clf=GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=0)
    pr=cross_val_predict(clf,X[m],y[m],cv=5,method='predict_proba')[:,1]
    from sklearn.metrics import roc_auc_score; print(f'CV AUC stage1 vs nanobanana: {roc_auc_score(y[m],pr):.3f}')
    for d in sorted(set(dom[m])): sel=dom[m]==d; print(f'  {d:11s} n={sel.sum():5d} доля "nb-like" (p>0.5): {(pr[sel]>0.5).mean():.1%}  p median {np.median(pr[sel]):.2f}')
    clf.fit(X[m],y[m]); pt=clf.predict_proba(X[~m])[:,1]
    print(f'TEST n={len(pt)}: nb-like (p>0.5) {(pt>0.5).mean():.1%}, p>0.8 {(pt>0.8).mean():.1%}, p median {np.median(pt):.2f}; гистограмма p:',np.histogram(pt,bins=[0,.1,.3,.5,.7,.9,1])[0].tolist())
    names=['w','h','aspect','side1024','qt_mean','qt0','qt1','hp_med','hp_gt2','flat_hp_med','flat_hp_gt1','flat_hp_gt3','flat_hp_std','flat_frac']
    imp=sorted(zip(clf.feature_importances_,names),reverse=True)[:6]; print('важность признаков:',[(n,round(v,2)) for v,n in imp])
    # без метаданных: только шумовые признаки (устойчивее к "случайному совпадению" размеров)
    idx=[7,8,9,10,11,12,13]; clf2=GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=0)
    pr2=cross_val_predict(clf2,X[m][:,idx],y[m],cv=5,method='predict_proba')[:,1]; print(f'только шумовые признаки: CV AUC {roc_auc_score(y[m],pr2):.3f}')
    clf2.fit(X[m][:,idx],y[m]); pt2=clf2.predict_proba(X[~m][:,idx])[:,1]; print(f'  TEST nb-like по шуму (p>0.5): {(pt2>0.5).mean():.1%}, p>0.8 {(pt2>0.8).mean():.1%}')
    for d in sorted(set(dom[m])): sel=dom[m]==d; print(f'  {d:11s} nb-like по шуму: {(pr2[sel]>0.5).mean():.1%}')
