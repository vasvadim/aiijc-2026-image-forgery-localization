#!/usr/bin/env python
"""tune_threshold.py — подбор порога t и минимальной площади a_min по AIC на val-tune (D-05).

Вход: <probs_dir> с PNG вероятностей (uint8, prob·255, исходный размер) от
`predict.py --role val-tune --save-probs`. Для каждой картинки один раз считаются Dice и площадь
при каждом t; a_min применяется аналитически (площадь < a_min → маска пустая: Dice=0, не FA).
Выбор — центр самого широкого плато (AIC ≥ max − tol), не точка максимума. Пишет threshold.json.

    python src/tune_threshold.py --probs-dir runs/.../val-tune/probs --out runs/.../threshold.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_dir, load_gt, load_split, wandb_update_summary  # noqa: E402
from metric import FA_AREA, MAIN_POOLS, aic, select_plateau  # noqa: E402

T_GRID = np.round(np.arange(0.10, 0.96, 0.05), 2)
AMIN_GRID = np.array([0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10])


def _one(args):
    probs_dir, root, row, t_grid = args
    prob = np.asarray(Image.open(Path(probs_dir) / f'{Path(row["chng_img_path"]).stem}.png'))
    gt = load_gt(Path(root) / row['gt_path']) if row['gt_path'] else None
    if gt is not None and gt.shape != prob.shape:
        raise ValueError(f'{row["chng_img_path"]}: prob {prob.shape} vs gt {gt.shape}')
    is_pos = bool(gt is not None and gt.any())
    dices, areas = [], []
    gsum = int(gt.sum()) if is_pos else 0
    for t in t_grid:
        pred = prob >= int(round(t * 255))
        areas.append(float(pred.mean()))
        dices.append(2.0 * int((pred & gt).sum()) / (int(pred.sum()) + gsum + 1e-6) if is_pos else 0.0)
    return is_pos, row['neg_pool'], np.array(dices), np.array(areas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs-dir', required=True)
    ap.add_argument('--version', default='v1')
    ap.add_argument('--role', default='val-tune')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tol', type=float, default=0.002)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    df = load_split(args.version, args.role)
    root = data_dir()
    with Pool(args.workers) as pool:
        res = pool.map(_one, [(args.probs_dir, root, r, T_GRID) for r in df.to_dict('records')], chunksize=16)
    is_pos = np.array([r[0] for r in res]); pools = np.array([r[1] for r in res])
    D = np.stack([r[2] for r in res]); A = np.stack([r[3] for r in res])       # [n, T]

    grid = np.zeros((len(T_GRID), len(AMIN_GRID)))
    detail = {}
    for j, a_min in enumerate(AMIN_GRID):
        keep = A >= a_min if a_min > 0 else np.ones_like(A, bool)               # маска не обнулена
        dice_pos = (D[is_pos] * keep[is_pos]).mean(axis=0)                       # [T]
        fa = (A >= FA_AREA) & keep
        fprs = {str(p): fa[(~is_pos) & (pools == p)].mean(axis=0) for p in np.unique(pools[~is_pos])}
        main = [fprs[p] for p in MAIN_POOLS if p in fprs]
        fpr_main = np.mean(main, axis=0) if main else np.zeros(len(T_GRID))
        for i in range(len(T_GRID)):
            grid[i, j] = aic(float(dice_pos[i]), float(fpr_main[i]))
            detail[(i, j)] = dict(dice_pos=float(dice_pos[i]), fpr_main=float(fpr_main[i]),
                                  fpr={p: float(v[i]) for p, v in fprs.items()})
    i, j = select_plateau(grid, tol=args.tol)
    imax = np.unravel_index(int(np.argmax(grid)), grid.shape)
    chosen = dict(t=float(T_GRID[i]), a_min=float(AMIN_GRID[j]), aic=float(grid[i, j]), **detail[(i, j)],
                  argmax=dict(t=float(T_GRID[imax[0]]), a_min=float(AMIN_GRID[imax[1]]), aic=float(grid.max())),
                  tol=args.tol, role=args.role, version=args.version, n=len(df),
                  t_grid=T_GRID.tolist(), amin_grid=AMIN_GRID.tolist(), grid=grid.round(5).tolist())
    Path(args.out).write_text(json.dumps(chosen, indent=1), encoding='utf-8')

    print('AIC grid (rows t, cols a_min):')
    print('   t  | ' + ' '.join(f'{a:6.3f}' for a in AMIN_GRID))
    for ii, t in enumerate(T_GRID):
        mark = '*' if ii == i else ' '
        print(f'{mark}{t:.2f} | ' + ' '.join(f'{v:6.4f}' for v in grid[ii]))
    print(f'chosen (plateau center): t={chosen["t"]} a_min={chosen["a_min"]} AIC={chosen["aic"]:.4f} '
          f'dice={chosen["dice_pos"]:.4f} fpr_main={chosen["fpr_main"]:.4f} fpr={chosen["fpr"]}')
    print(f'argmax: {chosen["argmax"]}')
    if i in (0, len(T_GRID) - 1) or j in (0, len(AMIN_GRID) - 1):
        print('WARNING: выбранная точка на краю сетки — расширь T_GRID/AMIN_GRID или проверь модель')
    print(f'saved {args.out}')
    run_dir = Path(args.probs_dir).resolve().parents[1]
    if (run_dir / 'wandb_id.txt').exists():
        wandb_update_summary(run_dir, {'threshold/t': chosen['t'], 'threshold/a_min': chosen['a_min'],
                                       'threshold/val_tune_aic': chosen['aic'], 'threshold/val_tune_dice': chosen['dice_pos'],
                                       'threshold/val_tune_fpr_main': chosen['fpr_main'], 'threshold/grid': grid.round(4).tolist()})


if __name__ == '__main__':
    main()
