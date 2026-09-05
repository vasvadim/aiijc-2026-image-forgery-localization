# -*- coding: utf-8 -*-
"""Качество JPEG теста по квант-таблицам (те же оценки, что для доменов train) + размеры. Только метаданные."""
import os, glob, json
from collections import Counter
import numpy as np
from PIL import Image
STD = np.array([16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99], float)
def q_est(qt):
    # обратная формула libjpeg: scale = qt/std → q
    sc = np.median(np.array(qt[:64], float) / STD) * 100
    return 100 - sc / 2 if sc <= 100 else 5000 / sc
def stats(files):
    qs, qt1, sizes, side, fmts = [], [], [], [], Counter()
    for f in files:
        im = Image.open(f); fmts[im.format] += 1
        w, h = im.size; sizes.append((w, h)); side.append(max(w, h))
        qt = getattr(im, 'quantization', None)
        if qt:
            qs.append(q_est(qt[0])); qt1.append(float(np.mean(qt[0])))
    qs = np.array(qs); side = np.array(side)
    ws = np.array([s[0] for s in sizes]); hs = np.array([s[1] for s in sizes])
    return dict(n=len(files), fmt=dict(fmts), q_hist={str(k): int(v) for k, v in sorted(Counter(np.round(qs).astype(int)).items())},
                q_pct={p: float(np.percentile(qs, p)) for p in (5, 10, 25, 50, 75, 90, 95)}, qt_mean_pct={p: float(np.percentile(qt1, p)) for p in (10, 50, 90)},
                side1024=float((side == 1024).mean()), side_pct={p: int(np.percentile(side, p)) for p in (5, 25, 50, 75, 95)},
                w_med=float(np.median(ws)), h_med=float(np.median(hs)), common_sizes=[f'{k[0]}x{k[1]}:{v}' for k, v in Counter(sizes).most_common(8)])
test = sorted(glob.glob(os.path.expanduser('~/aic2026/data/test_stage1/test_stage1_img/*')))
print('TEST', json.dumps(stats(test), indent=1))
# для сравнения: те же оценки на train по доменам (выборка 300) — чтобы шкала q была одна
import pandas as pd, random
random.seed(0)
sp = pd.read_csv(os.path.expanduser('~/aic2026/code_vadim/splits/stage1_v1.csv.gz'))
for d, g in sp.groupby('domain'):
    fs = [os.path.expanduser('~/aic2026/data/') + p for p in random.sample(list(g.chng_img_path), min(300, len(g)))]
    s = stats(fs); print(d, 'q_pct', {k: round(v) for k, v in s['q_pct'].items()}, 'side1024', round(s['side1024'], 2))
nb = pd.read_csv(os.path.expanduser('~/aic2026/data/nanobanana/pico_nanobanana.csv'))
for col in ('orgl_img_path', 'chng_img_path'):
    fs = [os.path.expanduser('~/aic2026/data/') + p for p in random.sample(list(nb[col]), 300)]
    fs = [f if os.path.exists(f) else f.replace('/data/', '/data/nanobanana/') for f in fs]
    s = stats(fs); print('nb', col, 'q_pct', {k: round(v) for k, v in s['q_pct'].items()}, 'side1024', round(s['side1024'], 2))
