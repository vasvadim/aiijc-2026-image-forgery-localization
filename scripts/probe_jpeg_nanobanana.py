# -*- coding: utf-8 -*-
"""Зонд: (1) как модель реагирует на nanobanana-правки и их оригиналы; (2) как diff-маски nb согласуются с предсказанием;
(3) чувствительность к JPEG-пережатию на val-tune позитивах (Dxx q100 → q85/q75). Тест не трогаем."""
import os, sys, json, io
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch
from PIL import Image
sys.path.insert(0, os.path.expanduser('~/aic2026/code_vadim/src'))
from common import data_dir, load_split, load_rgb, load_gt, normalize, parse_size
from predict import load_model
from metric import dice as dice_fn

ckpt, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
dev = torch.device('cuda')
model, ck = load_model(ckpt, dev); model.eval()
W, H = parse_size(ck['img_size'])
D = data_dir()
def P(p):
    for base in (D, D / 'stage1', D / 'nanobanana'):
        if (base / p).exists(): return base / p
    return D / p
T, TAU = 0.45, 0.8

@torch.no_grad()
def run(img_uint8):
    x = cv2.resize(img_uint8, (W, H), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(normalize(x))[None].to(dev)
    with torch.autocast('cuda', dtype=torch.float16):
        o = model(x)
    m, c = (o[0], o[1]) if isinstance(o, (tuple, list)) else (o, None)
    prob = torch.sigmoid(m.float())[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (img_uint8.shape[1], img_uint8.shape[0]), interpolation=cv2.INTER_LINEAR)
    return prob, (float(torch.sigmoid(c.float())[0, 0]) if c is not None else None)

def jpeg(img, q):
    buf = io.BytesIO(); Image.fromarray(img).save(buf, 'JPEG', quality=q); buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))

def diff_mask(a, b, thr=40, k=7):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(2) > thr
    d = d.astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((k * 3, k * 3), np.uint8))
    return d > 0

rep = {}
# ---------- (1)+(2) nanobanana
nb = pd.read_csv(D / 'nanobanana/pico_nanobanana.csv').fillna('').sample(300, random_state=0)
rows = []
for _, r in nb.iterrows():
    src = load_rgb(P(r.orgl_img_path)); chg = load_rgb(P(r.chng_img_path))
    p_c, c_c = run(chg); p_s, c_s = run(src)
    pred_c = p_c >= T; pred_s = p_s >= T
    dm = diff_mask(src, chg)
    rows.append(dict(p_img_chng=c_c, p_img_src=c_s, area_chng=float(pred_c.mean()), area_src=float(pred_s.mean()),
                     diff_area=float(dm.mean()), dice_pred_vs_diff=float(dice_fn(pred_c, dm)) if dm.any() else np.nan,
                     fa_chng=bool(pred_c.mean() >= 0.01 and (c_c is None or c_c >= TAU)), fa_src=bool(pred_s.mean() >= 0.01 and (c_s is None or c_s >= TAU))))
df = pd.DataFrame(rows); df.to_csv(out / 'nb_probe.csv', index=False)
rep['nb'] = dict(n=len(df), detect_rate_chng=float(df.fa_chng.mean()), fpr_src=float(df.fa_src.mean()),
                 p_img_chng_q=[float(x) for x in df.p_img_chng.quantile([.1, .25, .5, .75, .9])], p_img_src_q=[float(x) for x in df.p_img_src.quantile([.1, .25, .5, .75, .9])],
                 area_chng_q=[float(x) for x in df.area_chng.quantile([.1, .25, .5, .75, .9])], diff_area_q=[float(x) for x in df.diff_area.quantile([.1, .25, .5, .75, .9])],
                 dice_pred_vs_diff_median=float(df.dice_pred_vs_diff.median()), dice_pred_vs_diff_gt05=float((df.dice_pred_vs_diff > 0.5).mean()))
print(json.dumps(rep, indent=1), flush=True)

# ---------- (3) JPEG-чувствительность по доменам val-tune
sp = load_split('v1', 'val-tune'); sp = sp[sp.is_neg == 0]
res = {}
for dom in ['Dxx', 'ids8', 'coco', 'ids9', 'raise', 'inp', 'openimages']:
    sub = sp[sp.domain == dom].sample(min(80, (sp.domain == dom).sum()), random_state=0)
    d0, d85, d75, d95 = [], [], [], []
    for _, r in sub.iterrows():
        img = load_rgb(P(r.chng_img_path)); gt = load_gt(P(r.gt_path))
        for q, acc in ((None, d0), (95, d95), (85, d85), (75, d75)):
            im = img if q is None else jpeg(img, q)
            p, c = run(im); pred = p >= T
            if c is not None and c < TAU: pred[:] = False
            acc.append(float(dice_fn(pred, gt)))
    res[dom] = dict(n=len(sub), orig=float(np.mean(d0)), q95=float(np.mean(d95)), q85=float(np.mean(d85)), q75=float(np.mean(d75)))
    print(dom, res[dom], flush=True)
rep['jpeg_sensitivity'] = res
# негативы: FPR при пережатии
neg = load_split('v1', 'val-tune'); neg = neg[neg.is_neg == 1].sample(300, random_state=0)
fa = {None: 0, 95: 0, 85: 0, 75: 0}
for _, r in neg.iterrows():
    img = load_rgb(P(r.chng_img_path))
    for q in fa:
        p, c = run(img if q is None else jpeg(img, q)); pred = p >= T
        if c is not None and c < TAU: pred[:] = False
        fa[q] += pred.mean() >= 0.01
rep['jpeg_fpr'] = {str(q): fa[q] / len(neg) for q in fa}
print(json.dumps(rep, indent=1), flush=True)
(out / 'probe.json').write_text(json.dumps(rep, indent=1))
