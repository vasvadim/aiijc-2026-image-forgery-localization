#!/usr/bin/env python
"""metric.py — официальная локальная метрика AIC Score (VALIDATION.md §1, §6; D-04).

Повторяет логику организаторов один в один и работает ТОЛЬКО с PNG-файлами в формате сабмита,
в исходном разрешении картинок. Прокси-метрики внутри обучения живут в другом месте.

Формулы:
    Dice_i   = 2·|P∩G| / (|P| + |G| + 1e-6)          per image, среднее по positive (GT непустая)
    FA_j     = 1[ |P| / (H·W) >= 0.01 ]               по negative (GT пустая)
    AIC      = 2·D·(1−F) / (D + (1−F))

Использование (holdout):
    python src/metric.py --submission runs/<login>/<exp>/val-tune/submission.csv \
        --split splits/stage1_v1.csv.gz --role val-tune [--out metrics.json]
    python src/metric.py --submission .../control/submission.csv --split splits/nanobanana_control_v1.csv
    python src/metric.py --selftest

`submission.csv`: колонки img_path (= chng_img_path из split, относительно AIC_DATA_DIR) и
prediction_path (относительно папки, где лежит submission.csv). Все строки split выбранной роли
обязаны присутствовать. Любой PNG не в режиме L, со значениями кроме {0,255} или не того размера —
ошибка на весь прогон (на платформе это, скорее всего, ноль за сабмит).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

VERSION = 'v1'
FA_AREA = 0.01                 # порог площади ложной тревоги (условие задачи)
GT_THRESH = 128                # основная бинаризация GT (как бейзлайн организаторов)
GT_THRESH_ALT = 0              # альтернатива для полосы неопределённости
AREA_BUCKETS = [(0, 0.01, '<1%'), (0.01, 0.05, '1-5%'), (0.05, 0.20, '5-20%'), (0.20, 1.01, '>20%')]
MAIN_POOLS = ('neg_train', 'neg_src')   # FPR_main = среднее по этим пулам с равными весами (D-01)


# ----------------------------------------------------------------------------- чистые функции
def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (pred.sum() + gt.sum() + 1e-6))


def is_false_alarm(pred: np.ndarray) -> bool:
    return bool(pred.mean() >= FA_AREA)


def aic(dice_pos: float, fpr_neg: float) -> float:
    a, b = dice_pos, 1.0 - fpr_neg
    return 0.0 if a + b == 0 else float(2 * a * b / (a + b))


def _longest_run_center(ok: np.ndarray) -> int:
    """Центр самого длинного непрерывного отрезка True; если True нет — 0."""
    best_start, best_len, start = 0, 0, None
    for i, v in enumerate(list(ok) + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    return best_start + (best_len - 1) // 2


def select_plateau(scores: np.ndarray, tol: float = 0.002) -> tuple[int, ...]:
    """Индекс центра самого широкого плато (значения >= max − tol) на 1D/2D сетке (D-05).
    2D: строка (первая ось) с наибольшим числом «хороших» ячеек (при равенстве — с большим
    средним по ним), в ней — центр самого длинного отрезка «хороших» ячеек."""
    scores = np.asarray(scores, dtype=float)
    ok = scores >= scores.max() - tol
    if scores.ndim == 1:
        return (_longest_run_center(ok),)
    cnt = ok.sum(axis=1)
    mean_ok = np.where(cnt > 0, (scores * ok).sum(axis=1) / np.maximum(cnt, 1), -np.inf)
    row = int(np.lexsort((mean_ok, cnt))[-1])
    return (row, _longest_run_center(ok[row]))


# ----------------------------------------------------------------------------- чтение файлов
def _load_binary_png(path: str, expect_hw: tuple[int, int]) -> np.ndarray:
    im = Image.open(path)
    if im.mode != 'L':
        raise ValueError(f'{path}: режим {im.mode}, ожидается L')
    a = np.asarray(im)
    if a.shape != expect_hw:
        raise ValueError(f'{path}: размер {a.shape}, ожидается {expect_hw}')
    u = np.unique(a)
    if not set(u.tolist()) <= {0, 255}:
        raise ValueError(f'{path}: значения {u[:10].tolist()}, ожидаются только 0/255')
    return a > 0


def _load_gt(path: str, expect_hw: tuple[int, int]) -> np.ndarray:
    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape != expect_hw:
        raise ValueError(f'GT {path}: размер {m.shape}, ожидается {expect_hw}')
    return m


def _score_row(args):
    data_dir, sub_dir, row = args
    img_path = os.path.join(data_dir, row['chng_img_path'])
    with Image.open(img_path) as im:
        W, H = im.size
    pred = _load_binary_png(os.path.join(sub_dir, row['prediction_path']), (H, W))
    rec = dict(chng_img_path=row['chng_img_path'], domain=row['domain'], generator=row['generator'],
               neg_pool=row['neg_pool'], pred_area=float(pred.mean()), fa=is_false_alarm(pred))
    if row['gt_path']:
        g = _load_gt(os.path.join(data_dir, row['gt_path']), (H, W))
        g_main, g_alt = g > GT_THRESH, g > GT_THRESH_ALT
        rec.update(gt_area=float(g_main.mean()), pos_main=bool(g_main.any()), pos_alt=bool(g_alt.any()),
                   dice_main=dice(pred, g_main), dice_alt=dice(pred, g_alt))
    else:
        rec.update(gt_area=0.0, pos_main=False, pos_alt=False, dice_main=np.nan, dice_alt=np.nan)
    return rec


# ----------------------------------------------------------------------------- агрегация
def _fpr(recs: pd.DataFrame, pos_col: str) -> dict[str, tuple[float, int]]:
    neg = recs[~recs[pos_col]]
    out = {}
    for pool, d in neg.groupby('neg_pool'):
        out[pool or 'gt_empty'] = (float(d['fa'].mean()), len(d))
    return out


def aggregate(recs: pd.DataFrame) -> dict:
    """Все числа отчёта из per-image записей. Positive/negative определяет GT (при GT_THRESH)."""
    res = {'version': VERSION, 'n': len(recs)}
    for tag, pos_col, dice_col in (('main', 'pos_main', 'dice_main'), ('alt', 'pos_alt', 'dice_alt')):
        pos = recs[recs[pos_col]]
        fprs = _fpr(recs, pos_col)
        main_pools = [fprs[p][0] for p in MAIN_POOLS if p in fprs]
        fpr_main = float(np.mean(main_pools)) if main_pools else (np.mean([v[0] for v in fprs.values()]) if fprs else 0.0)
        d = float(pos[dice_col].mean()) if len(pos) else float('nan')
        res[tag] = dict(gt_thresh=GT_THRESH if tag == 'main' else GT_THRESH_ALT, n_pos=len(pos), dice_pos=d,
                        fpr=fprs, fpr_main=fpr_main, aic_main=aic(d, fpr_main),
                        aic_by_pool={p: aic(d, f) for p, (f, _) in fprs.items()})
    pos = recs[recs['pos_main']]
    res['empty_pred_on_pos'] = float((pos['pred_area'] == 0).mean()) if len(pos) else float('nan')
    res['dice_by_gt_area'] = {}
    for lo, hi, name in AREA_BUCKETS:
        d = pos[(pos['gt_area'] >= lo) & (pos['gt_area'] < hi)]
        res['dice_by_gt_area'][name] = (float(d['dice_main'].mean()) if len(d) else float('nan'), len(d))
    res['by_domain'], res['by_generator'] = {}, {}
    for key, store in (('domain', res['by_domain']), ('generator', res['by_generator'])):
        for val, d in recs.groupby(key):
            p, n = d[d['pos_main']], d[~d['pos_main']]
            store[val] = dict(n_pos=len(p), dice=float(p['dice_main'].mean()) if len(p) else float('nan'),
                              n_neg=len(n), fpr=float(n['fa'].mean()) if len(n) else float('nan'))
    return res


def format_report(res: dict, role: str) -> str:
    m, a = res['main'], res['alt']
    L = [f'=== AIC local metric {res["version"]} | role={role} | n={res["n"]} ===',
         f'Dice_pos (GT>{m["gt_thresh"]}): {m["dice_pos"]:.4f}  n_pos={m["n_pos"]}    '
         f'[GT>{a["gt_thresh"]}: {a["dice_pos"]:.4f}, n_pos={a["n_pos"]}]']
    for pool, (f, n) in m['fpr'].items():
        L.append(f'FPR {pool:10s}: {f:.4f}  (n={n})   AIC with this pool: {m["aic_by_pool"][pool]:.4f}')
    L.append(f'FPR_main (mean of {", ".join(p for p in MAIN_POOLS if p in m["fpr"]) or "all"}): {m["fpr_main"]:.4f}')
    L.append(f'>>> AIC_main: {m["aic_main"]:.4f}    [GT>{a["gt_thresh"]}: {a["aic_main"]:.4f}]')
    L.append(f'empty predictions on positives: {res["empty_pred_on_pos"]:.4f}')
    L.append('Dice by GT area: ' + '  '.join(f'{k}: {v:.4f} (n={n})' for k, (v, n) in res['dice_by_gt_area'].items()))
    for key in ('by_domain', 'by_generator'):
        L.append(f'-- {key[3:]}: ' + ' | '.join(
            f'{k}: dice={v["dice"]:.3f}/{v["n_pos"]} fpr={v["fpr"]:.3f}/{v["n_neg"]}' for k, v in res[key].items()))
    return '\n'.join(L)


# ----------------------------------------------------------------------------- CLI
def load_split(path: str, role: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if role == 'proxy':
        df = df[df['in_proxy'] == '1']
    elif role:
        df = df[df['role'] == role]
    if df.empty:
        sys.exit(f'в {path} нет строк с ролью {role}')
    return df.reset_index(drop=True)


def evaluate(submission_csv: str, split_csv: str, role: str | None, data_dir: str, workers: int = 16) -> dict:
    split = load_split(split_csv, role)
    sub = pd.read_csv(submission_csv, dtype=str, keep_default_na=False)
    sub['img_path'] = sub['img_path'].str.strip().str.replace('\\', '/', regex=False)
    missing = set(split['chng_img_path']) - set(sub['img_path'])
    if missing:
        sys.exit(f'в submission нет {len(missing)} строк split, например {sorted(missing)[:3]}')
    merged = split.merge(sub, left_on='chng_img_path', right_on='img_path', how='left')
    sub_dir = str(Path(submission_csv).resolve().parent)
    jobs = [(data_dir, sub_dir, r) for r in merged.to_dict('records')]
    with Pool(workers) as pool:
        recs = pool.map(_score_row, jobs, chunksize=32)
    recs = pd.DataFrame(recs)
    res = aggregate(recs)
    res['role'], res['submission'], res['split'] = role, submission_csv, split_csv
    return res


def selftest() -> None:
    """Синтетическая проверка формул и проверок формата на временных файлах."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    (tmp / 'img').mkdir(); (tmp / 'gt').mkdir(); (tmp / 'pred').mkdir()
    H, W = 100, 200
    rows, subs = [], []

    def put(name, gt, pred, pool=''):
        Image.new('RGB', (W, H)).save(tmp / 'img' / f'{name}.jpg')
        gt_path = ''
        if gt is not None:
            Image.fromarray(gt).save(tmp / 'gt' / f'{name}.png'); gt_path = f'gt/{name}.png'
        Image.fromarray(pred).save(tmp / 'pred' / f'{name}.png')
        rows.append(dict(chng_img_path=f'img/{name}.jpg', gt_path=gt_path, domain='t', generator='t',
                         neg_pool=pool, role='val-tune', in_proxy='0'))
        subs.append(dict(img_path=f'img/{name}.jpg', prediction_path=f'pred/{name}.png'))

    z = np.zeros((H, W), np.uint8)
    g1 = z.copy(); g1[:50, :] = 255          # GT верхняя половина
    p1 = z.copy(); p1[:25, :] = 255          # pred четверть → dice = 2·25/(25+50) = 0.6667
    put('a', g1, p1)
    put('b', g1, g1.copy())                  # dice 1
    put('c', g1, z.copy())                   # пустой pred → dice 0
    g_soft = z.copy(); g_soft[:50, :] = 255; g_soft[50:60, :] = 100   # серый ободок: >128 нет, >0 есть
    put('d', g_soft, g1.copy())              # main: dice 1; alt: 2·50/(50+60)=0.9091
    n1 = z.copy(); n1[0, :100] = 255         # 100 px = 0.5 % площади → не FA
    put('n1', z.copy(), n1, 'neg_train')
    n2 = z.copy(); n2[0, :] = 255            # 200 px = ровно 1 % → FA (граница >=)
    put('n2', z.copy(), n2, 'neg_train')
    put('s1', None, z.copy(), 'neg_src')     # src-негатив без GT, пустой pred
    put('s2', None, n2.copy(), 'neg_src')    # FA
    pd.DataFrame(rows).to_csv(tmp / 'split.csv', index=False)
    pd.DataFrame(subs).to_csv(tmp / 'submission.csv', index=False)
    res = evaluate(str(tmp / 'submission.csv'), str(tmp / 'split.csv'), 'val-tune', str(tmp), workers=2)
    m, a = res['main'], res['alt']
    exp_dice = (2 / 3 + 1 + 0 + 1) / 4
    assert abs(m['dice_pos'] - exp_dice) < 1e-6, m['dice_pos']
    assert abs(a["dice_pos"] - (2 * 25 / (25 + 50) + 1 + 0 + 2 * 50 / (50 + 60)) / 4) < 1e-6, a["dice_pos"]
    assert m['fpr']['neg_train'] == (0.5, 2) and m['fpr']['neg_src'] == (0.5, 2), m['fpr']
    assert abs(m['aic_main'] - aic(exp_dice, 0.5)) < 1e-9
    assert res['empty_pred_on_pos'] == 0.25
    # проверки формата
    bad = np.zeros((H, W), np.uint8); bad[0, 0] = 7
    Image.fromarray(bad).save(tmp / 'pred' / 'a.png')
    try:
        evaluate(str(tmp / 'submission.csv'), str(tmp / 'split.csv'), 'val-tune', str(tmp), workers=2)
        raise AssertionError('значение 7 не поймано')
    except ValueError:
        pass
    Image.fromarray(np.zeros((H, W + 1), np.uint8)).save(tmp / 'pred' / 'a.png')
    try:
        evaluate(str(tmp / 'submission.csv'), str(tmp / 'split.csv'), 'val-tune', str(tmp), workers=2)
        raise AssertionError('размер не пойман')
    except ValueError:
        pass
    assert select_plateau(np.array([0.1, 0.5, 0.9, 0.9, 0.9, 0.2, 0.9])) == (3,)
    assert select_plateau(np.array([[0.1, 0.9, 0.9, 0.9], [0.9, 0.1, 0.1, 0.1]])) == (0, 2)
    print(format_report(res, 'selftest'))
    print('SELFTEST OK')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--submission')
    ap.add_argument('--split')
    ap.add_argument('--role', default=None, help='train | val-tune | val-report | proxy | control; пусто = весь файл')
    ap.add_argument('--data-dir', default=os.environ.get('AIC_DATA_DIR'))
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', help='metrics.json')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not (args.submission and args.split and args.data_dir):
        sys.exit('нужны --submission, --split и AIC_DATA_DIR/--data-dir')
    res = evaluate(args.submission, args.split, args.role, args.data_dir, args.workers)
    print(format_report(res, args.role or 'all'))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
        print(f'saved {args.out}')


if __name__ == '__main__':
    main()
