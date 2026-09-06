import os,sys,cv2,numpy as np,pandas as pd
sys.path.insert(0,os.path.expanduser('~/aic2026/code_vadim/src')); sys.path.insert(0,os.path.expanduser('~/aic2026/logs'))
from common import data_dir
from vad_diff_calib import P
D=data_dir(); OUT=os.path.expanduser('~/aic2026/logs/vad_diffcal')
def mask(a,b,t,o,c,minfrac):
    d=(np.abs(a.astype(np.int16)-b.astype(np.int16)).max(2)>t).astype(np.uint8)
    d=cv2.morphologyEx(d,cv2.MORPH_OPEN,np.ones((o,o),np.uint8)); d=cv2.morphologyEx(d,cv2.MORPH_CLOSE,np.ones((c,c),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(d,8); keep=np.zeros_like(d)
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA]>=minfrac*d.size: keep[lab==i]=1
    return keep
RECIPES={'A t25 o5c15 min0.1%':(25,5,15,0.001),'B t30 o7c21 min0.2%':(30,7,21,0.002),'C t40 o7c21 min0.3%':(40,7,21,0.003)}
nb=pd.read_csv(D/'nanobanana/pico_nanobanana.csv').fillna('')
s=nb.sample(16,random_state=11); rows=[]
for r in s.itertuples():
    a=cv2.imread(str(P(r.orgl_img_path))); b=cv2.imread(str(P(r.chng_img_path))); tiles=[cv2.resize(a,(300,int(300*a.shape[0]/a.shape[1])))]
    for name,(t,o,c,mf) in RECIPES.items():
        m=mask(a,b,t,o,c,mf); ov=b.copy(); ov[m>0]=(0.5*ov[m>0]+np.array([0,0,127])).astype(np.uint8)
        cn,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov,cn,-1,(0,255,0),2)
        ov=cv2.resize(ov,(300,int(300*ov.shape[0]/ov.shape[1]))); cv2.putText(ov,f'{name.split()[0]} {m.mean()*100:.1f}%',(4,18),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2); tiles.append(ov)
    h=max(t.shape[0] for t in tiles); can=np.zeros((h,1220,3),np.uint8)
    for i,t in enumerate(tiles): can[:t.shape[0],i*305:i*305+300]=t
    cv2.putText(can,str(r.prompt)[:75],(4,h-6),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1); rows.append(can)
cv2.imwrite(f'{OUT}/nb_recipes_cmp.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,80])
# распределения площадей на 400 парах
s2=nb.sample(400,random_state=5); areas={k:[] for k in RECIPES}
for r in s2.itertuples():
    a=cv2.imread(str(P(r.orgl_img_path))); b=cv2.imread(str(P(r.chng_img_path)))
    if a is None or b is None or a.shape!=b.shape: continue
    for k,(t,o,c,mf) in RECIPES.items(): areas[k].append(mask(a,b,t,o,c,mf).mean())
for k,v in areas.items():
    v=np.array(v); print(f'{k}: median {np.median(v):.3f} | <1%: {(v<0.01).mean():.0%} 1-5%: {((v>=0.01)&(v<0.05)).mean():.0%} 5-20%: {((v>=0.05)&(v<0.2)).mean():.0%} 20-50%: {((v>=0.2)&(v<0.5)).mean():.0%} >50%: {(v>=0.5).mean():.0%}')
print('ok')
