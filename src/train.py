#!/usr/bin/env python
"""train.py — обучение сегментатора на split v1 с тремя критериями выбора эпохи (D-08).

Каждую эпоху считаются ВСЕ три критерия на val-tune / proxy-подмножестве:
    loss   — val loss на всём val-tune;
    proxy  — Dice per image + FPR (по пулам, равные веса) на тензорах в рабочем разрешении,
             t=0.5, без a_min, только строки in_proxy=1 (секунды);
    full   — официальная metric.py на PNG в исходном разрешении, t=0.5, a_min=0, те же proxy-строки
             (~1 мин; --full-every N, чтобы считать реже, 0 — не считать).
Сохраняются last.pth, best_loss.pth, best_proxy.pth, best_full.pth; best.pth = копия по --select-by.
Лог эпох: <run_dir>/log.jsonl. Конфиг: <run_dir>/config.json.

Одна GPU:
    CUDA_VISIBLE_DEVICES=6 python src/train.py --run-name 20260902_baseline_r34_256 \
        --encoder resnet34 --img-size 256 --epochs 15 --batch-size 32 --lr 1e-4 --loss bce --amp
Несколько GPU (DDP, тот же скрипт; --batch-size — на КАЖДУЮ карту, lr не масштабируется автоматически):
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 src/train.py --run-name ... --batch-size 32 ...
Валидация и full-метрика выполняются на rank 0 (остальные ждут на barrier); чекпоинты/логи пишет rank 0.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_dir, load_split, runs_dir, seed_everything, split_path  # noqa: E402
from data import SegDataset  # noqa: E402
from metric import MAIN_POOLS, aic  # noqa: E402
from predict import predict_to_files  # noqa: E402

FA_AREA = 0.01


# ----------------------------------------------------------------------------- DDP helpers
def ddp_setup() -> tuple[int, int, int]:
    """Возвращает (rank, world_size, local_rank); без torchrun — (0, 1, 0)."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group('nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier():
    if is_dist():
        dist.barrier()


def all_reduce_mean(x: float, device) -> float:
    if not is_dist():
        return x
    t = torch.tensor([x], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


# ----------------------------------------------------------------------------- model / loss
def build_model(encoder: str, weights: str | None = 'imagenet') -> nn.Module:
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1, activation=None)


class BceDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = dice_weight

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
        return bce + self.w * dice.mean()


def make_loss(name: str) -> nn.Module:
    if name == 'bce':
        return nn.BCEWithLogitsLoss()
    if name == 'bce_dice':
        return BceDiceLoss()
    raise ValueError(name)


# ----------------------------------------------------------------------------- validation
def proxy_metrics(dice_list, area_list, df_proxy: pd.DataFrame) -> dict:
    """Dice per image по позитивам (is_neg==0), FPR по пулам среди негативов; как metric.aggregate, но на тензорах."""
    d = np.array(dice_list)
    a = np.array(area_list)
    pos = df_proxy['is_neg'].values == 0
    dice_pos = float(d[pos].mean()) if pos.any() else float('nan')
    fpr = {}
    for pool in df_proxy.loc[~pos, 'neg_pool'].unique():
        sel = (~pos) & (df_proxy['neg_pool'].values == pool)
        fpr[pool] = float((a[sel] >= FA_AREA).mean())
    main = [fpr[p] for p in MAIN_POOLS if p in fpr]
    fpr_main = float(np.mean(main)) if main else float(np.mean(list(fpr.values()))) if fpr else 0.0
    return dict(dice_pos=dice_pos, fpr=fpr, fpr_main=fpr_main, aic=aic(dice_pos, fpr_main))


@torch.no_grad()
def validate(model, loader, criterion, device, df_val: pd.DataFrame, amp: bool):
    """Один проход по val-tune: val loss на всех строках + proxy-метрики на in_proxy==1."""
    model.eval()
    total, n = 0.0, 0
    prox_idx, prox_dice, prox_area = [], [], []
    for batch in tqdm(loader, desc='val', leave=False, disable=None):
        x = batch['image'].to(device, non_blocking=True)
        y = batch['mask'].to(device, non_blocking=True)
        with torch.autocast('cuda', enabled=amp):
            logits = model(x)
        total += criterion(logits.float(), y).item() * x.shape[0]
        n += x.shape[0]
        idx = batch['idx'].numpy()
        inp = df_val['in_proxy'].values[idx] == 1
        if inp.any():
            p = (torch.sigmoid(logits.float()) > 0.5)[inp]
            g = (y > 0.5)[inp]
            inter = (p & g).sum(dim=(1, 2, 3)).float()
            dice = (2 * inter / (p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3)) + 1e-6)).cpu().numpy()
            area = p.float().mean(dim=(1, 2, 3)).cpu().numpy()
            prox_idx.extend(idx[inp].tolist()); prox_dice.extend(dice.tolist()); prox_area.extend(area.tolist())
    order = np.argsort(prox_idx)
    df_proxy = df_val.iloc[np.array(prox_idx)[order]]
    pm = proxy_metrics(np.array(prox_dice)[order], np.array(prox_area)[order], df_proxy)
    return total / max(n, 1), pm


def full_metric(model, df_proxy, args, device, run_dir: Path, epoch: int) -> dict:
    """Официальная метрика на PNG (t=0.5, a_min=0) на proxy-строках; PNG удаляются после подсчёта."""
    from metric import evaluate
    out = run_dir / f'full_ep{epoch:02d}'
    predict_to_files(model, df_proxy, out, img_size=args.img_size, t=0.5, a_min=0.0,
                     batch_size=args.batch_size, workers=args.workers, device=device, amp=args.amp,
                     root=data_dir(), path_col='chng_img_path', save_probs=False)
    res = evaluate(str(out / 'submission.csv'), str(args.split_file), 'proxy', str(data_dir()), workers=args.workers)
    shutil.rmtree(out, ignore_errors=True)
    m = res['main']
    return dict(dice_pos=m['dice_pos'], fpr={k: v[0] for k, v in m['fpr'].items()}, fpr_main=m['fpr_main'], aic=m['aic_main'])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--version', default='v1')
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--img-size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=32, help='на одну GPU')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--loss', default='bce', choices=['bce', 'bce_dice'])
    ap.add_argument('--scheduler', default='none', choices=['none', 'cosine'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8, help='DataLoader-воркеров на процесс')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--channels-last', action='store_true')
    ap.add_argument('--select-by', default='proxy', choices=['loss', 'proxy', 'full'])
    ap.add_argument('--full-every', type=int, default=1, help='считать full-метрику каждые N эпох; 0 — никогда')
    ap.add_argument('--limit', type=int, default=0, help='отладка: ограничить число train-строк')
    args = ap.parse_args()

    rank, world, local_rank = ddp_setup()
    main_proc = rank == 0
    seed_everything(args.seed)
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    run_dir = runs_dir() / args.run_name
    args.split_file = split_path(args.version)
    if main_proc:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'config.json').write_text(json.dumps({**vars(args), 'split_file': str(args.split_file),
                                                         'world_size': world}, indent=2, ensure_ascii=False))
    barrier()

    df_train = load_split(args.version, 'train')
    if args.limit:
        df_train = df_train.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)
    df_val = load_split(args.version, 'val-tune')
    df_proxy = df_val[df_val['in_proxy'] == 1].reset_index(drop=True)
    if main_proc:
        print(f'train={len(df_train)} val-tune={len(df_val)} proxy={len(df_proxy)} world={world} run_dir={run_dir}')

    ds_train = SegDataset(df_train, args.img_size, train=True)
    sampler = DistributedSampler(ds_train, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                          num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_val = None
    if main_proc:
        dl_val = DataLoader(SegDataset(df_val, args.img_size, train=False), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    model = build_model(args.encoder).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    ddp_model = DDP(model, device_ids=[local_rank]) if world > 1 else model
    criterion = make_loss(args.loss)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.scheduler == 'cosine' else None
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    best = {'loss': float('inf'), 'proxy': -1.0, 'full': -1.0}
    log_f = open(run_dir / 'log.jsonl', 'a', encoding='utf-8') if main_proc else None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        ddp_model.train()
        tr_loss, nb = 0.0, 0
        for batch in tqdm(dl_train, desc=f'train ep{epoch}', leave=False, disable=None if main_proc else True):
            x = batch['image'].to(device, non_blocking=True)
            y = batch['mask'].to(device, non_blocking=True)
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=args.amp):
                logits = ddp_model(x)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item(); nb += 1
        if scheduler:
            scheduler.step()
        tr_loss = all_reduce_mean(tr_loss / max(nb, 1), device)
        t_train = time.time() - t0

        if main_proc:
            va_loss, pm = validate(model, dl_val, criterion, device, df_val, args.amp)
            fm = None
            if args.full_every and (epoch % args.full_every == 0 or epoch == args.epochs):
                fm = full_metric(model, df_proxy, args, device, run_dir, epoch)
            rec = dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss, proxy=pm, full=fm,
                       lr=optimizer.param_groups[0]['lr'], time_s=round(time.time() - t0), train_time_s=round(t_train))
            log_f.write(json.dumps(rec, ensure_ascii=False) + '\n'); log_f.flush()
            print(f'ep{epoch} train={tr_loss:.4f} val={va_loss:.4f} | proxy dice={pm["dice_pos"]:.4f} '
                  f'fpr={pm["fpr_main"]:.4f} aic={pm["aic"]:.4f}'
                  + (f' | full dice={fm["dice_pos"]:.4f} fpr={fm["fpr_main"]:.4f} aic={fm["aic"]:.4f}' if fm else '')
                  + f' | {rec["time_s"]}s (train {rec["train_time_s"]}s)', flush=True)
            state = dict(epoch=epoch, model=model.state_dict(), encoder=args.encoder, img_size=args.img_size, metrics=rec)
            torch.save(state, run_dir / 'last.pth')
            if va_loss < best['loss']:
                best['loss'] = va_loss; torch.save(state, run_dir / 'best_loss.pth')
            if pm['aic'] > best['proxy']:
                best['proxy'] = pm['aic']; torch.save(state, run_dir / 'best_proxy.pth')
            if fm and fm['aic'] > best['full']:
                best['full'] = fm['aic']; torch.save(state, run_dir / 'best_full.pth')
        barrier()

    if main_proc:
        log_f.close()
        src = run_dir / f'best_{args.select_by}.pth'
        if src.exists():
            shutil.copyfile(src, run_dir / 'best.pth')
        (run_dir / 'best.json').write_text(json.dumps({'select_by': args.select_by, **best}, indent=2))
        print('done; best:', best)
    if is_dist():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
