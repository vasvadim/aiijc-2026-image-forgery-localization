# -*- coding: utf-8 -*-
"""Пункт 3: сколько даёт свой порог (t, a_min, tau) на каждый домен против одного общего.
Оракул (домен известен) на val-tune probs: in-sample и честно — 2-fold cross-fit по сценам (group_id).
    python oracle_domain_thr.py <val-tune out dir с probs/ и img_probs.csv> <out.json>
"""
import sys, os, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.expanduser('~/aic2026/code_vadim/src'))
from common import data_dir, load_split

OUT_DIR = Path(sys.argv[1]); RES = Path(sys.argv[2])
T = np.round(np.arange(0.20, 0.71, 0.05), 2)
AMIN = [0.0, 0.005, 0.01, 0.02]
TAU = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
D = data_dir()

def one(args):
    stem, gt_path = args
    p = cv2.imread(str(OUT_DIR / 'probs' / f'{stem}.png'), 0)
    if p is None: return None
    gt = (cv2.imread(str(D / gt_path), 0) > 128) if isinstance(gt_path, str) and gt_path else None
    dice, area = np.zeros(len(T)), np.zeros(len(T))
    for i, t in enumerate(T):
        pred = p >= int(round(t * 255))
        area[i] = pred.mean()
        if gt is not None and gt.any():
            inter = np.logical_and(pred, gt).sum(); dice[i] = 2 * inter / (pred.sum() + gt.sum())
    return stem, dice, area

def main():
    df = load_split('v1', 'val-tune').copy()
    df['stem'] = df.chng_img_path.map(lambda s: Path(s).stem)
    ip = pd.read_csv(OUT_DIR / 'img_probs.csv')[['stem', 'p_img']]; ip['stem'] = ip.stem.astype(str)
    df = df.merge(ip, on='stem', how='left')
    df['dom'] = np.where(df.is_neg == 1, df.neg_pool + ':' + df.domain.fillna('?'), df.domain)
    with Pool(32) as pool:
        res = pool.map(one, [(s, g if isinstance(g, str) else '') for s, g in zip(df.stem, df.gt_path)], chunksize=64)
    res = {r[0]: r for r in res if r is not None}
    df = df[df.stem.isin(res)].reset_index(drop=True)
    DICE = np.stack([res[s][1] for s in df.stem]); AREA = np.stack([res[s][2] for s in df.stem])
    pimg = df.p_img.fillna(1.0).values; is_pos = (df.is_neg == 0).values
    pool_id = df.neg_pool.fillna('').values
    dom = df.domain.fillna('?').values
    doms = sorted(set(dom))
    print('rows', len(df), 'pos', is_pos.sum(), 'domains', doms, flush=True)

    def score(cfg, mask=None):
        """cfg: dict domain -> (ti, a_min, tau). Возвращает (aic, dice, fpr)."""
        ti = np.array([cfg[d][0] for d in dom]); am = np.array([cfg[d][1] for d in dom]); tau = np.array([cfg[d][2] for d in dom])
        idx = np.arange(len(df))
        area = AREA[idx, ti]; dice = DICE[idx, ti]
        empty = (pimg < tau) | (area < am)
        dice = np.where(empty, 0.0, dice); fa = (~empty) & (area >= 0.01)
        m = np.ones(len(df), bool) if mask is None else mask
        dpos = dice[m & is_pos].mean()
        fprs = [fa[m & (pool_id == p)].mean() for p in ('neg_train', 'neg_src')]
        fpr = float(np.mean(fprs))
        aic = 2 * dpos * (1 - fpr) / (dpos + 1 - fpr) if dpos + 1 - fpr > 0 else 0
        return aic, dpos, fpr

    grid = [(i, a, tau) for i in range(len(T)) for a in AMIN for tau in TAU]

    def tune_global(mask):
        best = max(grid, key=lambda g: score({d: g for d in doms}, mask)[0])
        return {d: best for d in doms}

    def tune_domain(mask, start, passes=3):
        cfg = dict(start)
        for _ in range(passes):
            for d in doms:
                bestg, bests = cfg[d], score(cfg, mask)[0]
                for g in grid:
                    c2 = dict(cfg); c2[d] = g; s = score(c2, mask)[0]
                    if s > bests + 1e-9: bestg, bests = g, s
                cfg[d] = bestg
        return cfg

    fmt = lambda g: dict(t=float(T[g[0]]), a_min=g[1], tau=g[2])
    out = {}
    # in-sample оракул
    g0 = tune_global(None); s0 = score(g0)
    gd = tune_domain(None, g0); sd = score(gd)
    out['insample'] = dict(global_cfg=fmt(next(iter(g0.values()))), global_aic=s0, domain_cfg={d: fmt(gd[d]) for d in doms}, domain_aic=sd)
    print('IN-SAMPLE global', fmt(next(iter(g0.values()))), s0, '\n          domain', sd, {d: fmt(gd[d]) for d in doms}, flush=True)
    # 2-fold cross-fit по сценам
    h = np.array([hash(g) % 2 for g in df.group_id.astype(str)])
    folds = []
    for k in (0, 1):
        tr, te = h != k, h == k
        gk = tune_global(tr); dk = tune_domain(tr, gk)
        folds.append(dict(global_heldout=score(gk, te), domain_heldout=score(dk, te)))
        print(f'FOLD {k}: global held-out {score(gk, te)}  domain held-out {score(dk, te)}', flush=True)
    out['crossfit'] = folds
    out['crossfit_mean'] = dict(global_aic=float(np.mean([f['global_heldout'][0] for f in folds])), domain_aic=float(np.mean([f['domain_heldout'][0] for f in folds])))
    print('CROSS-FIT mean', out['crossfit_mean'], flush=True)
    RES.write_text(json.dumps(out, indent=1, default=float))

if __name__ == '__main__':
    main()
