import os,sys,json,cv2,numpy as np,pandas as pd
from multiprocessing import Pool
sys.path.insert(0,os.path.expanduser('~/aic2026/code_vadim/src'))
from common import data_dir, load_split
D=data_dir(); OUT=os.path.expanduser('~/aic2026/logs/vad_diffcal'); os.makedirs(OUT,exist_ok=True)
def P(p):
    for b in (D, D/'stage1', D/'nanobanana'):
        if (b/p).exists(): return b/p
    return D/p
TS=[10,15,20,25,30,40,50]; MORPH={'none':None,'o3c7':(3,7),'o5c15':(5,15),'o7c21':(7,21)}
def diffmask(a,b,t,m):
    d=(np.abs(a.astype(np.int16)-b.astype(np.int16)).max(2)>t).astype(np.uint8)
    if m: d=cv2.morphologyEx(d,cv2.MORPH_OPEN,np.ones((m[0],m[0]),np.uint8)); d=cv2.morphologyEx(d,cv2.MORPH_CLOSE,np.ones((m[1],m[1]),np.uint8))
    return d>0
def jpeg(img,q):
    ok,buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,q]); return cv2.imdecode(buf,1)
def dice(p,g): return 2*(p&g).sum()/max(1,p.sum()+g.sum())
def one(row):
    a=cv2.imread(str(P(row[0]))); b=cv2.imread(str(P(row[1]))); g=cv2.imread(str(P(row[2])),0)>128
    if a is None or b is None or a.shape!=b.shape: return None
    res={}
    for mode,(aa,bb) in {'clean':(a,b),'noised':(jpeg(a,88),jpeg(b,92))}.items():
        noise=float(np.abs(aa.astype(np.int16)-bb.astype(np.int16)).max(2)[~g].mean()) if (~g).any() else 0
        for t in TS:
            for mn,m in MORPH.items(): res[(mode,t,mn)]=dice(diffmask(aa,bb,t,m),g)
        res[(mode,'noise')]=noise
    return res
if __name__=='__main__':
    sp=load_split('v1','train'); pairs=sp[(sp.is_neg==0)&sp.orgl_img_path.notna()].sample(600,random_state=0)
    rows=[(r.orgl_img_path,r.chng_img_path,r.gt_path) for r in pairs.itertuples()]
    with Pool(32) as pool: rs=[r for r in pool.map(one,rows) if r]
    print('stage1 pairs same shape:',len(rs))
    summ={}
    for mode in ('clean','noised'):
        print(f'== {mode}: noise outside GT mean|diff| = {np.mean([r[(mode,"noise")] for r in rs]):.1f}')
        best=None
        for t in TS:
            line=[]
            for mn in MORPH:
                v=float(np.mean([r[(mode,t,mn)] for r in rs])); line.append(f'{mn}={v:.3f}'); summ[f'{mode}/t{t}/{mn}']=v
                if best is None or v>best[0]: best=(v,t,mn)
            print(f'  t={t:2d}: '+'  '.join(line))
        print(f'  BEST {mode}: t={best[1]} morph={best[2]} dice={best[0]:.3f}'); summ[f'best_{mode}']=dict(t=best[1],morph=best[2],dice=best[0])
    json.dump(summ,open(f'{OUT}/calib.json','w'),indent=1)
    # применяем рецепт noised к nanobanana
    bt,bm=summ['best_noised']['t'],MORPH[summ['best_noised']['morph']]
    nb=pd.read_csv(D/'nanobanana/pico_nanobanana.csv').fillna('')
    s=nb.sample(500,random_state=1); areas=[]
    for r in s.itertuples():
        a=cv2.imread(str(P(r.orgl_img_path))); b=cv2.imread(str(P(r.chng_img_path)))
        if a is None or b is None or a.shape!=b.shape: continue
        areas.append(float(diffmask(a,b,bt,bm).mean()))
    areas=np.array(areas); hist={'<1%':float((areas<0.01).mean()),'1-5%':float(((areas>=0.01)&(areas<0.05)).mean()),'5-20%':float(((areas>=0.05)&(areas<0.2)).mean()),'20-50%':float(((areas>=0.2)&(areas<0.5)).mean()),'50-80%':float(((areas>=0.5)&(areas<0.8)).mean()),'>80%':float((areas>=0.8).mean())}
    print('nb area distribution with recipe:',{k:f'{v:.0%}' for k,v in hist.items()},'median',f'{np.median(areas):.3f}')
    json.dump({'recipe':dict(t=bt,morph=summ['best_noised']['morph']),'nb_area_hist':hist,'nb_area_median':float(np.median(areas))},open(f'{OUT}/nb_recipe.json','w'),indent=1)
    # мозаика: 16 nb пар с маской по рецепту
    tiles=[]
    for r in nb.sample(16,random_state=11).itertuples():
        a=cv2.imread(str(P(r.orgl_img_path))); b=cv2.imread(str(P(r.chng_img_path)))
        m=diffmask(a,b,bt,bm).astype(np.uint8); over=b.copy(); over[m>0]=(0.5*over[m>0]+np.array([0,0,127])).astype(np.uint8)
        c,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(over,c,-1,(0,255,0),2)
        sm=cv2.resize(a,(240,int(240*a.shape[0]/a.shape[1]))); ov=cv2.resize(over,(480,int(480*over.shape[0]/over.shape[1])))
        can=np.zeros((360,724,3),np.uint8); can[:min(360,sm.shape[0]),:240]=sm[:360]; can[:min(360,ov.shape[0]),244:724]=ov[:360]
        cv2.putText(can,f'area={m.mean()*100:.1f}%',(248,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.putText(can,str(r.prompt)[:62],(4,352),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1); tiles.append(can)
    cv2.imwrite(f'{OUT}/nb_recipe_mosaic.jpg',np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,16,2)]),[cv2.IMWRITE_JPEG_QUALITY,82]); print('mosaic ok')
