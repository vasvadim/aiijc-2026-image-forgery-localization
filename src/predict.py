#!/usr/bin/env python
"""predict.py — инференс чекпоинта в формате сабмита (predictions/*.png + submission.csv).

Роли split (val-tune / val-report / proxy / control) или тест (--test). Постобработка: порог t,
минимальная площадь a_min (если доля белого < a_min — маска обнуляется). Опционально сохраняет
вероятности (uint8 PNG, prob·255, исходный размер) в <out>/probs/ для tune_threshold.py.
После инференса по роли split автоматически считает metric.py и пишет metrics.json.

Примеры:
    python src/predict.py --checkpoint runs/.../best.pth --role val-tune --out runs/.../val-tune --save-probs
    python src/predict.py --checkpoint runs/.../best.pth --role val-report --out runs/.../val-report --t 0.45 --a-min 0.01
    python src/predict.py --checkpoint runs/.../best.pth --test --out runs/.../test --t 0.45 --a-min 0.01 --zip
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_dir, load_split, load_test_df, save_binary_png, split_path  # noqa: E402
from data import SegDataset  # noqa: E402

cv2.setNumThreads(0)


def postprocess(prob: np.ndarray, t: float, a_min: float) -> np.ndarray:
    pred = prob >= t
    if a_min > 0 and pred.mean() < a_min:
        pred[:] = False
    return pred


def _write(args):
    prob, h, w, t, a_min, pred_path, prob_path = args
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    save_binary_png(postprocess(prob, t, a_min), pred_path)
    if prob_path:
        cv2.imwrite(str(prob_path), np.clip(prob * 255 + 0.5, 0, 255).astype(np.uint8))


@torch.no_grad()
def predict_to_files(model, df, out: Path, img_size: int, t: float, a_min: float, batch_size: int, workers: int,
                     device, amp: bool, root: Path, path_col: str, save_probs: bool) -> Path:
    """Пишет <out>/predictions/<stem>_pred.png, <out>/submission.csv (img_path = df[path_col])."""
    out = Path(out)
    pred_dir = out / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    prob_dir = out / 'probs'
    if save_probs:
        prob_dir.mkdir(exist_ok=True)
    ds = SegDataset(df, img_size, train=False, root=root, path_col=path_col)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    model.eval()
    rows = []
    with ThreadPoolExecutor(max_workers=max(4, workers)) as ex:
        futs = []
        for batch in tqdm(dl, desc='predict', leave=False, disable=None):
            x = batch['image'].to(device, non_blocking=True)
            with torch.autocast('cuda', enabled=amp):
                logits = model(x)
            probs = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
            for i in range(probs.shape[0]):
                idx = int(batch['idx'][i]); h = int(batch['orig_h'][i]); w = int(batch['orig_w'][i])
                rel = df[path_col].iloc[idx]
                stem = Path(rel).stem
                pred_path = pred_dir / f'{stem}_pred.png'
                prob_path = (prob_dir / f'{stem}.png') if save_probs else None
                futs.append(ex.submit(_write, (probs[i], h, w, t, a_min, pred_path, prob_path)))
                rows.append({'img_path': rel, 'prediction_path': f'predictions/{stem}_pred.png'})
        for f in futs:
            f.result()
    with open(out / 'submission.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['img_path', 'prediction_path'])
        wr.writeheader(); wr.writerows(rows)
    return out / 'submission.csv'


def load_model(ckpt_path: str, device):
    from train import build_model
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = build_model(ck['encoder'], weights=None)
    model.load_state_dict(ck['model'])
    return model.to(device), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--version', default='v1')
    ap.add_argument('--role', help='val-tune | val-report | proxy | control')
    ap.add_argument('--test', action='store_true', help='инференс теста организаторов')
    ap.add_argument('--out', required=True)
    ap.add_argument('--t', type=float, default=0.5)
    ap.add_argument('--a-min', type=float, default=0.0)
    ap.add_argument('--img-size', type=int, default=None, help='по умолчанию из чекпоинта')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--save-probs', action='store_true')
    ap.add_argument('--zip', action='store_true', help='собрать <out>/submission.zip (для теста)')
    args = ap.parse_args()
    if bool(args.role) == bool(args.test):
        sys.exit('укажи ровно одно: --role или --test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck = load_model(args.checkpoint, device)
    img_size = args.img_size or ck['img_size']
    out = Path(args.out)

    if args.test:
        df, root = load_test_df()
        path_col = 'img_path'
        split_file = None
    else:
        if args.role == 'control':
            df = load_split(str(split_path(args.version).with_name(f'nanobanana_control_{args.version}.csv')))
            split_file = split_path(args.version).with_name(f'nanobanana_control_{args.version}.csv')
        else:
            df = load_split(args.version, args.role)
            split_file = split_path(args.version)
        root, path_col = data_dir(), 'chng_img_path'

    sub = predict_to_files(model, df, out, img_size, args.t, args.a_min, args.batch_size, args.workers,
                           device, args.amp, root, path_col, args.save_probs)
    print(f'written {sub} ({len(df)} rows), t={args.t} a_min={args.a_min}')
    info = dict(checkpoint=args.checkpoint, epoch=ck.get('epoch'), t=args.t, a_min=args.a_min, img_size=img_size,
                role=args.role or 'test', n=len(df))
    if split_file is not None:
        from metric import evaluate, format_report
        res = evaluate(str(sub), str(split_file), None if args.role == 'control' else args.role, str(data_dir()), workers=args.workers)
        print(format_report(res, args.role))
        info['metrics'] = res
    (out / 'metrics.json').write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')
    if args.zip:
        zpath = out / 'submission.zip'
        with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(sub, 'submission.csv')
            for p in sorted((out / 'predictions').glob('*.png')):
                z.write(p, f'predictions/{p.name}')
        print(f'zip: {zpath} ({zpath.stat().st_size // 2**20} MiB)')


if __name__ == '__main__':
    main()
