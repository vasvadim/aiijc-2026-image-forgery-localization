# -*- coding: utf-8 -*-
"""Полный аудит выданных данных: что есть, что используем, что нет. Читает только train/nanobanana (тест не трогаем)."""
import os, re, sys, json, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import cv2

D = Path(os.environ.get('AIC_DATA_DIR', os.path.expanduser('~/aic2026/data')))
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
random.seed(0)
rep = {}

def P(p):
    for base in (D, D / 'stage1', D / 'nanobanana'):
        if (base / p).exists(): return base / p
    return D / p

# ---------- stage1 train.csv
tr = pd.read_csv(D / 'stage1/train.csv').fillna('')
rep['train_rows'] = len(tr)
rep['rows_with_src'] = int((tr.orgl_img_path != '').sum())
rep['unique_src'] = int(tr.loc[tr.orgl_img_path != '', 'orgl_img_path'].nunique())
src_dir = D / 'stage1/train/src'
src_files = sorted(os.listdir(src_dir))
rep['src_files_on_disk'] = len(src_files)
ref_src = set(Path(p).name for p in tr.orgl_img_path if p)
rep['src_files_not_in_csv'] = len(set(src_files) - ref_src)
rep['ext_src_orphans'] = Counter(Path(f).suffix for f in (set(src_files) - ref_src)).most_common()

# split
sp = pd.read_csv(os.path.expanduser('~/aic2026/code_vadim/splits/stage1_v1.csv.gz'))
rep['split_cols'] = list(sp.columns)
rep['split_roles'] = sp.groupby(['role', 'kind']).size().to_dict() if 'kind' in sp.columns else sp.role.value_counts().to_dict()
rep['neg_pool_by_role'] = sp.groupby(['role', 'neg_pool']).size().to_dict()
rep['src_scenes_train_role'] = int(sp[(sp.role == 'train') & sp.domain.isin(['coco', 'raise', 'openimages'])].group_id.nunique())
rep['split_role_domain'] = sp.groupby(['role', 'domain']).size().to_dict()
rep['rows_per_scene_src_domains'] = sp[sp.domain.isin(['coco', 'raise', 'openimages'])].groupby('group_id').size().describe().to_dict()

# ---------- размеры/формат изображений по доменам (сэмпл)
by_dom = defaultdict(list)
for p, d in zip(sp.chng_img_path, sp.domain):
    by_dom[d].append(p)
size_stats = {}
for d, lst in by_dom.items():
    s = random.sample(lst, min(300, len(lst)))
    ws, hs, fmts, qs = [], [], Counter(), []
    for p in s:
        try:
            im = Image.open(P(p))
            ws.append(im.size[0]); hs.append(im.size[1]); fmts[im.format] += 1
            q = im.info.get('quality')
            if im.format == 'JPEG':
                # оценка качества по квант-таблице (среднее значение первой таблицы)
                qt = getattr(im, 'quantization', None)
                if qt: qs.append(float(np.mean(qt[0])))
        except Exception as e:
            fmts['ERR'] += 1
    size_stats[d] = dict(n=len(lst), w_med=float(np.median(ws)), h_med=float(np.median(hs)), w_min=int(min(ws)), w_max=int(max(ws)), h_min=int(min(hs)), h_max=int(max(hs)),
                         side1024=float(np.mean([max(a, b) == 1024 for a, b in zip(ws, hs)])), fmt=dict(fmts), jpeg_qt_mean=float(np.mean(qs)) if qs else None, jpeg_qt_p10_p90=[float(np.percentile(qs, 10)), float(np.percentile(qs, 90))] if qs else None)
rep['size_by_domain'] = size_stats

# ---------- src vs chng: совпадает ли размер, доля отличия (для синтеза масок / copy-move)
pairs = tr[tr.orgl_img_path != ''].sample(200, random_state=0)
same, diff_area = 0, []
for _, r in pairs.iterrows():
    a = cv2.imread(str(P(r.orgl_img_path))); b = cv2.imread(str(P(r.chng_img_path)))
    if a is None or b is None: continue
    if a.shape == b.shape:
        same += 1
        m = cv2.imread(str(P(r.gt_path)), 0) > 128
        dd = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(2) > 25
        diff_area.append(dict(gt=float(m.mean()), diff=float(dd.mean()), diff_in_gt=float((dd & m).sum() / max(1, m.sum())), diff_out_gt=float((dd & ~m).sum() / max(1, (~m).sum()))))
rep['stage1_src_same_shape_frac'] = same / len(pairs)
rep['stage1_diff_vs_gt'] = {k: float(np.median([x[k] for x in diff_area])) for k in diff_area[0]} if diff_area else None

# ---------- nanobanana
nb = pd.read_csv(D / 'nanobanana/pico_nanobanana.csv').fillna('')
rep['nb_rows'] = len(nb); rep['nb_cols'] = list(nb.columns)
rep['nb_unique_src'] = int(nb.orgl_img_path.nunique()); rep['nb_unique_chng'] = int(nb.chng_img_path.nunique())
words = Counter()
for p in nb['prompt']:
    for w in re.findall(r'[a-zA-Z]+', str(p).lower()):
        if len(w) > 3: words[w] += 1
rep['nb_prompt_top_words'] = words.most_common(40)
rep['nb_prompt_examples'] = nb['prompt'].sample(15, random_state=0).tolist()
nbs = nb.sample(150, random_state=0)
same, stats, ws = 0, [], []
for _, r in nbs.iterrows():
    a = cv2.imread(str(P(r.orgl_img_path))); b = cv2.imread(str(P(r.chng_img_path)))
    if a is None or b is None: continue
    ws.append((a.shape[1], a.shape[0], b.shape[1], b.shape[0]))
    if a.shape == b.shape:
        same += 1
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(2)
        stats.append(dict(d25=float((d > 25).mean()), d40=float((d > 40).mean()), d60=float((d > 60).mean()), mean_abs=float(d.mean())))
rep['nb_same_shape_frac'] = same / len(nbs)
rep['nb_shape_examples'] = ws[:10]
rep['nb_diff_area_quantiles'] = {k: [float(np.percentile([s[k] for s in stats], q)) for q in (10, 25, 50, 75, 90)] for k in stats[0]} if stats else None
# global-change fraction (по «diff>25 занимает >70% кадра»)
rep['nb_global_change_frac'] = float(np.mean([s['d25'] > 0.7 for s in stats])) if stats else None
fm = Counter()
for p in nbs.chng_img_path[:60]:
    try: fm[Image.open(P(p)).format] += 1
    except Exception: fm['ERR'] += 1
rep['nb_chng_formats'] = dict(fm)
fm = Counter()
for p in nbs.orgl_img_path[:60]:
    try: fm[Image.open(P(p)).format] += 1
    except Exception: fm['ERR'] += 1
rep['nb_src_formats'] = dict(fm)

# ---------- маски stage1: доля с несколькими компонентами, формы
mc = []
for p in tr.sample(400, random_state=1).gt_path:
    m = (cv2.imread(str(P(p)), 0) > 128).astype(np.uint8)
    if m.sum() == 0: continue
    n, _ = cv2.connectedComponents(m)
    x, y, w, h = cv2.boundingRect(m)
    mc.append(dict(ncomp=n - 1, fill=float(m.sum() / max(1, w * h)), area=float(m.mean())))
rep['mask_ncomp_hist'] = dict(Counter(min(x['ncomp'], 5) for x in mc))
rep['mask_bbox_fill_median'] = float(np.median([x['fill'] for x in mc]))

def fix(o):
    if isinstance(o, dict): return {str(k): fix(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [fix(x) for x in o]
    return o
rep = fix(rep)
(OUT / 'eda_full.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
