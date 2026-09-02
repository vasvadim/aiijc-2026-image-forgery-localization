#!/usr/bin/env python
"""make_split.py — строит фиксированное разбиение train по спецификации VALIDATION.md §4
(решения D-02, D-03, D-06, D-09 в DECISIONS.md).

Запуск на кластере (≈1–2 мин на 32 воркерах, читает все 103k масок):

    export AIC_DATA_DIR=/home/vasilenkova/aic2026/data
    python src/make_split.py --version v1 --seed 42 --workers 32

Результат (в репо, папка splits/):
    stage1_<v>.csv.gz            все строки train + src-негативы holdout-сцен; колонки см. COLUMNS
    stage1_<v>_dropped.csv       битые пары (маска не совпала по размеру с картинкой)
    nanobanana_control_<v>.csv   контрольный пул neg_nb (оригиналы nanobanana)
    stage1_<v>_summary.md        таблица ролей/доменов/пулов — для глаз и для experiments.md

Версия v1 после генерации НЕ редактируется (D-09). Любое изменение логики → новая версия.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import os
import re
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO / 'splits'

N_FOLDS = 20
VAL_TUNE_FOLDS = (0, 1)
VAL_REPORT_FOLDS = (2,)
NEG_HOLDOUT_SHARE = 0.60        # доля neg_train-сцен, уходящих в holdout (D-06)
SRC_CAP = {'val-tune': 2000, 'val-report': 1000}   # кап на src-негативы по ролям (§3)
PROXY_POS = 1500                # позитивов в proxy-подмножестве (D-08)
NB_CONTROL = 2000               # оригиналов nanobanana в контрольном пуле
GT_THRESH = 128                 # бинаризация GT, как в бейзлайне организаторов (D-04)
RARE_STRATUM = 100              # страты меньше этого сливаются в <domain>_rare

COLUMNS = ['chng_img_path', 'gt_path', 'orgl_img_path', 'domain', 'generator', 'group_id',
           'fold', 'role', 'is_neg', 'neg_pool', 'in_proxy', 'H', 'W', 'gt_area']

HEX12 = re.compile(r'^[0-9a-f]{12}$')
GENERATORS = ['powerpaint', 'inpaintanything', 'removeanything', 'brushnet', 'hdpainter']
SIZE_TOKENS = {'small', 'medium', 'large'}


# ----------------------------------------------------------------------------- парсинг имён
def parse_name(path: str) -> dict:
    """Домен, генератор и id сцены (group_id) из имени файла. Правила — VALIDATION.md §2, §4."""
    stem = Path(path).stem
    toks = stem.split('_')
    if len(toks) == 1:                                   # 000196093.jpg
        return dict(domain='ids9', generator='unknown', group_id=f'ids9_{stem}')
    if HEX12.match(toks[0]):
        toks = toks[1:]
    fam = toks[0]
    gens = sorted((g for g in GENERATORS if g in stem), key=stem.index)
    gen = '+'.join(gens)
    if fam == 'coco':
        return dict(domain='coco', generator=gen or 'unknown', group_id=f'coco_{toks[1]}')
    if fam == 'raise':
        return dict(domain='raise', generator=gen or 'unknown', group_id=f'raise_{toks[1]}')
    if fam == 'openimages':
        return dict(domain='openimages', generator=gen or 'unknown', group_id=f'openimages_{toks[1]}')
    size_idx = next((i for i, t in enumerate(toks) if t in SIZE_TOKENS), None)
    if size_idx is not None:                             # D17_I_nat_0074_large_1_inpainted-0 / 3230101918_medium_3_inpainted-0
        if re.match(r'^D\d\d$', fam):
            # Dxx отбрасываем: одна сцена (I_nat_0074) может встречаться в разных Dxx — группируем консервативно
            return dict(domain='Dxx', generator=gen or 'inpainted', group_id='D_' + '_'.join(toks[1:size_idx]))
        return dict(domain='inp', generator=gen or 'inpainted', group_id='inp_' + '_'.join(toks[:size_idx]))
    if len(toks) == 1 and re.match(r'^\d{8}$', fam):     # d8aa3363436d_00030709.jpg — семейство ids8: и правки, и ВСЕ чистые кадры train
        return dict(domain='ids8', generator='unknown', group_id=f'ids8_{fam}')
    return dict(domain='other', generator=gen or 'unknown', group_id=f'other_{stem}')


# ----------------------------------------------------------------------------- чтение масок
def _read_pair(args):
    """Размер картинки, размер и площадь маски. Возвращает кортеж без исключений (ошибки в тексте)."""
    data_dir, img_rel, gt_rel = args
    try:
        with Image.open(os.path.join(data_dir, img_rel)) as im:
            W, H = im.size
        m = np.asarray(Image.open(os.path.join(data_dir, gt_rel)))
        if m.ndim == 3:
            m = m[:, :, 0]
        mh, mw = m.shape
        area = float((m > GT_THRESH).mean())
        return (H, W, mh, mw, area, '')
    except Exception as e:  # noqa: BLE001
        return (-1, -1, -1, -1, -1.0, repr(e))


def _read_size(args):
    data_dir, rel = args
    try:
        with Image.open(os.path.join(data_dir, rel)) as im:
            W, H = im.size
        return (H, W, '')
    except Exception as e:  # noqa: BLE001
        return (-1, -1, repr(e))


# ----------------------------------------------------------------------------- разбиение
def assign_positive_folds(df: pd.DataFrame, seed: int) -> pd.Series:
    """StratifiedGroupKFold по (domain, generator) с группами = сцены."""
    strat = df['domain'] + '/' + df['generator']
    counts = strat.value_counts()
    rare = counts[counts < RARE_STRATUM].index
    strat = strat.where(~strat.isin(rare), df['domain'] + '/rare')
    folds = pd.Series(-1, index=df.index, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # предупреждения о малых стратах — ожидаемы
        for k, (_, idx) in enumerate(sgkf.split(df, strat, groups=df['group_id'])):
            folds.iloc[idx] = k
    assert (folds >= 0).all()
    return folds


def assign_negative_folds(groups: list[str], seed: int) -> dict[str, int]:
    """60 % сцен-негативов в holdout (2:1 между val-tune и val-report), 40 % в train (D-06)."""
    rng = np.random.RandomState(seed + 1)
    groups = sorted(groups)
    rng.shuffle(groups)
    n_hold = int(round(len(groups) * NEG_HOLDOUT_SHARE))
    n_tune = int(round(n_hold * 2 / 3))
    out = {}
    for i, g in enumerate(groups[:n_tune]):
        out[g] = VAL_TUNE_FOLDS[i % len(VAL_TUNE_FOLDS)]
    for i, g in enumerate(groups[n_tune:n_hold]):
        out[g] = VAL_REPORT_FOLDS[i % len(VAL_REPORT_FOLDS)]
    train_folds = [f for f in range(N_FOLDS) if f not in VAL_TUNE_FOLDS + VAL_REPORT_FOLDS]
    for i, g in enumerate(groups[n_hold:]):
        out[g] = train_folds[i % len(train_folds)]
    return out


def role_of(fold: int) -> str:
    if fold in VAL_TUNE_FOLDS:
        return 'val-tune'
    if fold in VAL_REPORT_FOLDS:
        return 'val-report'
    return 'train'


def leakage_checks(df: pd.DataFrame) -> None:
    roles_per_group = df.groupby('group_id')['role'].nunique()
    bad = roles_per_group[roles_per_group > 1]
    assert bad.empty, f'сцены в нескольких ролях: {bad.index[:10].tolist()}'
    assert df['chng_img_path'].is_unique, 'дубли chng_img_path'
    train_src = set(df.loc[df['role'] == 'train', 'orgl_img_path'].dropna()) - {''}
    holdout_src_neg = set(df.loc[(df['neg_pool'] == 'neg_src'), 'chng_img_path'])
    assert not (train_src & holdout_src_neg), 'src train-сцены попал в holdout как негатив'
    holdout_src = set(df.loc[df['role'] != 'train', 'orgl_img_path'].dropna()) - {''}
    assert not (train_src & holdout_src), 'один src и в train, и в holdout'
    prox = df[df['in_proxy'] == 1]
    assert (prox['role'] == 'val-tune').all(), 'proxy вне val-tune'


def md_table(t: pd.DataFrame) -> str:
    """Markdown-таблица без зависимости от tabulate."""
    t = t.reset_index()
    head = '| ' + ' | '.join(str(c) for c in t.columns) + ' |'
    sep = '|' + '---|' * len(t.columns)
    body = ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in t.itertuples(index=False)]
    return '\n'.join([head, sep] + body)


def summary_md(df: pd.DataFrame, dropped: pd.DataFrame, nb: pd.DataFrame, version: str, seed: int) -> str:
    lines = [f'# Split stage1_{version} (seed={seed}, folds={N_FOLDS})', '']
    lines += ['## Роли', '', '| role | rows | positives | neg_train | neg_src | proxy |', '|---|---|---|---|---|---|']
    for role in ['train', 'val-tune', 'val-report']:
        d = df[df['role'] == role]
        lines.append(f"| {role} | {len(d)} | {int((d['is_neg'] == 0).sum())} | "
                     f"{int((d['neg_pool'] == 'neg_train').sum())} | {int((d['neg_pool'] == 'neg_src').sum())} | "
                     f"{int(d['in_proxy'].sum())} |")
    lines += ['', f'neg_nb (контроль, отдельный файл): {len(nb)}', f'dropped (маска ≠ размер картинки): {len(dropped)}', '']
    pos = df[df['is_neg'] == 0]
    lines += ['## Позитивы: домен × роль', '']
    tab = pd.crosstab(pos['domain'], pos['role'])
    tab['share_train'] = (tab['train'] / tab['train'].sum()).round(3)
    tab['share_val-tune'] = (tab['val-tune'] / tab['val-tune'].sum()).round(3)
    lines += [md_table(tab), '']
    lines += ['## Позитивы: генератор × роль', '', md_table(pd.crosstab(pos["generator"], pos["role"])), '']
    lines += ['## Площадь GT (позитивы, доля кадра)', '',
              pos.groupby('role')['gt_area'].describe(percentiles=[.05, .25, .5, .75, .95]).round(4).pipe(md_table), '']
    return '\n'.join(lines) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='v1')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--data-dir', default=os.environ.get('AIC_DATA_DIR'))
    ap.add_argument('--force', action='store_true', help='перезаписать существующую версию (только для отладки!)')
    args = ap.parse_args()
    if not args.data_dir:
        sys.exit('задай AIC_DATA_DIR или --data-dir')
    data_dir = args.data_dir
    out_main = SPLITS_DIR / f'stage1_{args.version}.csv.gz'
    if out_main.exists() and not args.force:
        sys.exit(f'{out_main} уже существует; версия неизменяема (D-09). Новая логика → новая версия.')
    SPLITS_DIR.mkdir(exist_ok=True)

    # 1. train.csv + парсинг имён
    df = pd.read_csv(os.path.join(data_dir, 'stage1', 'train.csv'), dtype=str).fillna('')
    for c in ('orgl_img_path', 'chng_img_path', 'gt_path'):
        df[c] = df[c].str.strip().str.replace('\\', '/', regex=False)
    parsed = pd.DataFrame([parse_name(p) for p in df['chng_img_path']])
    df = pd.concat([df, parsed], axis=1)
    print(f'rows={len(df)}  domains={df["domain"].value_counts().to_dict()}')

    # 2. читаем все маски: размер, площадь, пустота
    with Pool(args.workers) as pool:
        meta = pool.map(_read_pair, [(data_dir, a, b) for a, b in zip(df['chng_img_path'], df['gt_path'])], chunksize=256)
    meta = pd.DataFrame(meta, columns=['H', 'W', 'mh', 'mw', 'gt_area', 'err'])
    df = pd.concat([df, meta], axis=1)
    errs = df[df['err'] != '']
    assert errs.empty, f'ошибки чтения: {errs[["chng_img_path", "err"]].head()}'
    bad = (df['H'] != df['mh']) | (df['W'] != df['mw'])
    dropped = df.loc[bad, ['chng_img_path', 'gt_path', 'H', 'W', 'mh', 'mw']].copy()
    df = df.loc[~bad].drop(columns=['mh', 'mw', 'err']).reset_index(drop=True)
    df['is_neg'] = (df['gt_area'] == 0).astype(int)
    df.loc[df['is_neg'] == 1, 'generator'] = 'none'
    print(f'dropped={len(dropped)}  negatives={int(df["is_neg"].sum())}  '
          f'neg by domain={df[df.is_neg == 1]["domain"].value_counts().to_dict()}')

    # 3. фолды: позитивные сцены — StratifiedGroupKFold; чисто негативные сцены — отдельно
    grp_neg = df.groupby('group_id')['is_neg'].agg(['min', 'max'])
    all_neg_groups = grp_neg.index[grp_neg['min'] == 1].tolist()
    pos_mask = ~df['group_id'].isin(all_neg_groups)
    df['fold'] = -1
    df.loc[pos_mask, 'fold'] = assign_positive_folds(df.loc[pos_mask], args.seed).values
    neg_folds = assign_negative_folds(all_neg_groups, args.seed)
    df.loc[~pos_mask, 'fold'] = df.loc[~pos_mask, 'group_id'].map(neg_folds).values
    assert (df['fold'] >= 0).all()
    df['role'] = df['fold'].map(role_of)
    df['neg_pool'] = np.where(df['is_neg'] == 1, 'neg_train', '')

    # 4. src-негативы для holdout-сцен (D-01), с капом по ролям
    rng = np.random.RandomState(args.seed + 2)
    src_rows = []
    for role, cap in SRC_CAP.items():
        cand = df[(df['role'] == role) & (df['orgl_img_path'] != '')].drop_duplicates('orgl_img_path')
        take = cand.sample(n=min(cap, len(cand)), random_state=rng) if len(cand) > cap else cand
        for _, r in take.iterrows():
            src_rows.append(dict(chng_img_path=r['orgl_img_path'], gt_path='', orgl_img_path='',
                                 domain=r['domain'], generator='none', group_id=r['group_id'],
                                 fold=r['fold'], role=role, is_neg=1, neg_pool='neg_src', gt_area=0.0))
    src_df = pd.DataFrame(src_rows)
    with Pool(args.workers) as pool:
        sizes = pool.map(_read_size, [(data_dir, p) for p in src_df['chng_img_path']], chunksize=64)
    src_df['H'] = [s[0] for s in sizes]
    src_df['W'] = [s[1] for s in sizes]
    assert all(s[2] == '' for s in sizes), 'ошибка чтения src'
    df = pd.concat([df, src_df], ignore_index=True)

    # 5. proxy-подмножество: 1500 позитивов фолда 0 + все негативы val-tune (D-08)
    df['in_proxy'] = 0
    pos0 = df[(df['fold'] == 0) & (df['is_neg'] == 0)].sample(n=PROXY_POS, random_state=args.seed + 3).index
    df.loc[pos0, 'in_proxy'] = 1
    df.loc[(df['role'] == 'val-tune') & (df['is_neg'] == 1), 'in_proxy'] = 1

    # 6. контрольный пул nanobanana
    nb = pd.read_csv(os.path.join(data_dir, 'nanobanana', 'pico_nanobanana.csv'), dtype=str).fillna('')
    nb['orgl_img_path'] = nb['orgl_img_path'].str.strip().str.replace('\\', '/', regex=False)
    nb = nb.sample(n=min(NB_CONTROL, len(nb)), random_state=args.seed + 4)
    nb_df = pd.DataFrame(dict(chng_img_path=nb['orgl_img_path'].values, gt_path='', orgl_img_path='',
                              domain='nanobanana', generator='none',
                              group_id=['nb_' + Path(p).stem for p in nb['orgl_img_path']],
                              fold=-1, role='control', is_neg=1, neg_pool='neg_nb', in_proxy=0, gt_area=0.0))
    with Pool(args.workers) as pool:
        sizes = pool.map(_read_size, [(data_dir, p) for p in nb_df['chng_img_path']], chunksize=64)
    nb_df['H'] = [s[0] for s in sizes]
    nb_df['W'] = [s[1] for s in sizes]
    assert all(s[2] == '' for s in sizes), 'ошибка чтения nanobanana'

    # 7. проверки и запись
    df = df[COLUMNS].sort_values(['role', 'fold', 'chng_img_path']).reset_index(drop=True)
    df['gt_area'] = df['gt_area'].round(6)
    leakage_checks(df)
    with gzip.open(out_main, 'wt', encoding='utf-8', newline='') as f:
        df.to_csv(f, index=False)
    dropped.to_csv(SPLITS_DIR / f'stage1_{args.version}_dropped.csv', index=False)
    nb_df[COLUMNS].to_csv(SPLITS_DIR / f'nanobanana_control_{args.version}.csv', index=False)
    md = summary_md(df, dropped, nb_df, args.version, args.seed)
    (SPLITS_DIR / f'stage1_{args.version}_summary.md').write_text(md, encoding='utf-8')
    print(md)
    print(f'written: {out_main}')


if __name__ == '__main__':
    main()
