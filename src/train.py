#!/usr/bin/env python
"""train.py — обучение сегментатора на split v1 с тремя критериями выбора эпохи (D-08).

Каждую эпоху считаются ВСЕ три критерия на val-tune / proxy-подмножестве:
    loss   — val loss на всём val-tune;
    proxy  — Dice per image + FPR (по пулам, равные веса) на тензорах в рабочем разрешении,
             t=0.5, без a_min, только строки in_proxy=1 (секунды);
    full   — официальная metric.py на PNG в исходном разрешении, t=0.5, a_min=0, те же proxy-строки
             (~1 мин; --full-every N, чтобы считать реже, 0 — не считать).
Чекпоинты: last.pth — ПОЛНОЕ состояние для продолжения (модель, оптимизатор с моментами, scheduler,
GradScaler, эпоха, best, RNG); best_loss/best_proxy/best_full.pth — только веса; best.pth = по --select-by.
Продолжить прерванное обучение: --resume auto (берёт <run_dir>/last.pth) или --resume <path>.
Ранняя остановка: --patience N (критерий --select-by не улучшался N эпох подряд; 0 — выключить, D-12).
W&B: --wandb (по умолчанию включён, если установлен wandb и сделан `wandb login`); --no-wandb выключает.
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
import random
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMAGENET_MEAN, IMAGENET_STD, data_dir, load_split, parse_size, runs_dir, seed_everything, split_path  # noqa: E402
from data import SegDataset  # noqa: E402
from forensic import EXTRA_CHANNELS, ForensicUnet, adapt_state_dict, fuse_patch, patch_targets  # noqa: E402
from metric import MAIN_POOLS, aic  # noqa: E402
from predict import predict_to_files  # noqa: E402

FA_AREA = 0.01
AREA_BUCKETS = [(0, 0.01, 'lt1'), (0.01, 0.05, '1-5'), (0.05, 0.20, '5-20'), (0.20, 1.01, 'gt20')]
N_IMG_LOG = 8          # картинок в панели W&B (фиксированный набор из proxy)
STEP_LOG_EVERY = 50    # шагов между логами train/step_loss


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
def build_model(encoder: str, weights: str | None = 'imagenet', aux_head: bool = False, decoder_channels=None,
                extra_input: str = 'none', patch_head: bool = False) -> nn.Module:
    """UNet (ForensicUnet): forward → (mask_logits, img_logit | None, patch_logits | None); aux_head — image-level
    голова «есть ли правка»; extra_input/patch_head — см. forensic.py.
    decoder_channels — ширина UNet-декодера, по умолчанию smp (256,128,64,32,16); узкий (128,64,32,16,8) вдвое дешевле по FLOPs."""
    if encoder.startswith('dino:'):
        from dino_seg import DinoSeg
        return DinoSeg(encoder[5:], pretrained=weights is not None, aux_head=aux_head)
    aux = dict(classes=1, pooling='avg', dropout=0.2) if aux_head else None
    kw = dict(decoder_channels=tuple(decoder_channels)) if decoder_channels else {}
    return ForensicUnet(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1, activation=None, aux_params=aux,
                        extra_in=EXTRA_CHANNELS[extra_input], patch_head=patch_head, **kw)


def parse_decoder(v):
    """'128,64,32,16,8' → (128,64,32,16,8); None/'' → None (smp default)."""
    if not v:
        return None
    return tuple(int(x) for x in str(v).replace(' ', '').split(','))


def forward(model, x):
    """Единый вызов: (mask_logits, img_logit | None, patch_logits | None)."""
    out = model(x)
    if isinstance(out, (tuple, list)):
        return (out[0], out[1], out[2] if len(out) > 2 else None)
    return out, None, None


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


def edge_loss(logits, target, k: int = 5):
    """BCE только на полосе шириной ~k px вокруг контура GT (dilate − erode). RESEARCH/разбор ошибок:
    половина дефицита Dice — границы масок; BCE+Dice границы отдельно не штрафуют."""
    with torch.no_grad():
        dil = F.max_pool2d(target, k, stride=1, padding=k // 2)
        ero = -F.max_pool2d(-target, k, stride=1, padding=k // 2)
        band = (dil - ero) > 0
    if not band.any():
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[band], target[band])


def make_loss(name: str) -> nn.Module:
    if name == 'bce':
        return nn.BCEWithLogitsLoss()
    if name == 'bce_dice':
        return BceDiceLoss()
    raise ValueError(name)


# ----------------------------------------------------------------------------- validation
def proxy_metrics(dice, area, df_proxy: pd.DataFrame, img_prob=None, tau: float = 0.5) -> dict:
    """Dice per image по позитивам (is_neg==0), FPR по пулам среди негативов (как metric.aggregate, но на
    тензорах) + разрезы по доменам и площади GT, доля пустых прогнозов, медиана площади на негативах."""
    d = np.asarray(dice); a = np.asarray(area)
    pos = df_proxy['is_neg'].values == 0
    cls = None
    if img_prob is not None:  # гейтинг: p_img < tau → маска пустая (Dice=0 на позитиве, не FA на негативе)
        ip = np.asarray(img_prob)
        keep = ip >= tau
        cls = dict(acc=float(((ip >= 0.5) == pos).mean()), tpr=float((ip[pos] >= tau).mean()) if pos.any() else float('nan'),
                   tnr=float((ip[~pos] < tau).mean()) if (~pos).any() else float('nan'))
        d_ung, a_ung = d, a
        d = d * keep; a = a * keep
    dice_pos = float(d[pos].mean()) if pos.any() else float('nan')
    fpr = {}
    for pool in df_proxy.loc[~pos, 'neg_pool'].unique():
        sel = (~pos) & (df_proxy['neg_pool'].values == pool)
        fpr[pool] = float((a[sel] >= FA_AREA).mean())
    main = [fpr[p] for p in MAIN_POOLS if p in fpr]
    fpr_main = float(np.mean(main)) if main else float(np.mean(list(fpr.values()))) if fpr else 0.0
    out = dict(dice_pos=dice_pos, fpr=fpr, fpr_main=fpr_main, aic=aic(dice_pos, fpr_main),
               empty_pred_on_pos=float((a[pos] == 0).mean()) if pos.any() else float('nan'),
               neg_pred_area_median=float(np.median(a[~pos])) if (~pos).any() else float('nan'),
               dice_by_domain={}, dice_by_area={})
    dom = df_proxy['domain'].values
    for dname in np.unique(dom[pos]):
        out['dice_by_domain'][str(dname)] = float(d[pos & (dom == dname)].mean())
    ga = df_proxy['gt_area'].astype(float).values
    for lo, hi, name in AREA_BUCKETS:
        sel = pos & (ga >= lo) & (ga < hi)
        out['dice_by_area'][name] = float(d[sel].mean()) if sel.any() else float('nan')
    if cls is not None:
        out['cls'] = cls
        ung = proxy_metrics(d_ung, a_ung, df_proxy)
        out['aic_ungated'] = ung['aic']; out['dice_pos_ungated'] = ung['dice_pos']; out['fpr_main_ungated'] = ung['fpr_main']
    return out


def save_panel(panel: dict, df_val: pd.DataFrame, run_dir: Path, epoch: int) -> None:
    """Композит картинка + GT (зелёный контур) + предсказание (красная заливка) → <run_dir>/panel/epNN_k.png."""
    from PIL import Image
    out = run_dir / 'panel'; out.mkdir(exist_ok=True)
    for k, idx in enumerate(sorted(panel)):
        img, gt, pred = panel[idx]
        comp = img.astype(np.float32)
        comp[pred] = comp[pred] * 0.5 + np.array([255, 0, 0]) * 0.5
        er = gt.copy(); er[1:-1, 1:-1] = gt[1:-1, 1:-1] & gt[:-2, 1:-1] & gt[2:, 1:-1] & gt[1:-1, :-2] & gt[1:-1, 2:]
        edge = gt & ~er
        comp[edge] = [0, 255, 0]
        r = df_val.iloc[idx]
        tag = f"{r['domain']}_{'neg-' + r['neg_pool'] if r['is_neg'] else 'pos-' + str(round(float(r['gt_area']), 3))}"
        Image.fromarray(comp.astype(np.uint8)).save(out / f'ep{epoch:02d}_{k}_{tag}.png')


def _denorm(x: torch.Tensor) -> np.ndarray:
    img = x.cpu().numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _dice_area(prob, g):
    p = prob > 0.5
    inter = (p & g).sum(dim=(1, 2, 3)).float()
    dice = (2 * inter / (p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3)) + 1e-6)).cpu().numpy()
    return dice, p.float().mean(dim=(1, 2, 3)).cpu().numpy()


@torch.no_grad()
def validate(model, loader, criterion, device, df_val: pd.DataFrame, amp: bool, img_idx: set, patch_gamma: float = 0.0):
    """Один проход по val-tune: val loss на всех строках + proxy-метрики на in_proxy==1 + панель картинок.
    С patch-головой proxy считается по слитой вероятности (fuse_patch, γ=patch_gamma), а pm['nopatch'] —
    по чисто пиксельной, чтобы видеть вклад головы."""
    model.eval()
    total, n = 0.0, 0
    prox_idx, prox_dice, prox_area, prox_img, panel = [], [], [], [], {}
    pix_dice, pix_area = [], []
    for batch in tqdm(loader, desc='val', leave=False, disable=None):
        x = batch['image'].to(device, non_blocking=True)
        y = batch['mask'].to(device, non_blocking=True)
        with torch.autocast('cuda', enabled=amp):
            logits, img_logit, patch_logit = forward(model, x)
        total += criterion(logits.float(), y).item() * x.shape[0]
        n += x.shape[0]
        idx = batch['idx'].numpy()
        inp = df_val['in_proxy'].values[idx] == 1
        if inp.any():
            prob_pix = torch.sigmoid(logits.float())
            prob = fuse_patch(prob_pix, patch_logit, patch_gamma)
            g = (y > 0.5)[inp]
            dice, area = _dice_area(prob[inp], g)
            prox_idx.extend(idx[inp].tolist()); prox_dice.extend(dice.tolist()); prox_area.extend(area.tolist())
            if patch_logit is not None:
                d2, a2 = _dice_area(prob_pix[inp], g)
                pix_dice.extend(d2.tolist()); pix_area.extend(a2.tolist())
            if img_logit is not None:
                prox_img.extend(torch.sigmoid(img_logit.float())[inp, 0].cpu().numpy().tolist())
            for j in np.where(inp)[0]:
                if int(idx[j]) in img_idx:
                    panel[int(idx[j])] = (_denorm(x[j, :3]), (y[j, 0] > 0.5).cpu().numpy(), (prob[j, 0] > 0.5).cpu().numpy())
    order = np.argsort(prox_idx)
    df_proxy = df_val.iloc[np.array(prox_idx)[order]]
    img_prob = np.array(prox_img)[order] if prox_img else None
    pm = proxy_metrics(np.array(prox_dice)[order], np.array(prox_area)[order], df_proxy, img_prob=img_prob)
    if pix_dice:
        pp = proxy_metrics(np.array(pix_dice)[order], np.array(pix_area)[order], df_proxy, img_prob=img_prob)
        pm['nopatch'] = dict(aic=pp['aic'], dice_pos=pp['dice_pos'], fpr_main=pp['fpr_main'], dice_by_domain=pp['dice_by_domain'], dice_by_area=pp['dice_by_area'])
    return total / max(n, 1), pm, panel


def full_metric(model, df_proxy, args, device, run_dir: Path, epoch: int) -> dict:
    """Официальная метрика на PNG (t=0.5, a_min=0) на proxy-строках; PNG удаляются после подсчёта."""
    from metric import evaluate
    out = run_dir / f'full_ep{epoch:02d}'
    predict_to_files(model, df_proxy, out, img_size=args.img_size, t=0.5, a_min=0.0, tau=0.5 if args.aux_head else 0.0,
                     batch_size=args.batch_size, workers=args.workers, device=device, amp=args.amp,
                     root=data_dir(), path_col='chng_img_path', save_probs=False,
                     patch_gamma=args.patch_gamma if args.patch_head else 0.0, extra_input=args.extra_input)
    res = evaluate(str(out / 'submission.csv'), str(args.split_file), 'proxy', str(data_dir()), workers=args.workers)
    shutil.rmtree(out, ignore_errors=True)
    m = res['main']
    return dict(dice_pos=m['dice_pos'], fpr={k: v[0] for k, v in m['fpr'].items()}, fpr_main=m['fpr_main'], aic=m['aic_main'])


# ----------------------------------------------------------------------------- W&B
class WandbLogger:
    """Тонкая обёртка: молчит, если wandb не установлен / не залогинен / выключен флагом."""

    def __init__(self, enabled: bool, project: str, run_name: str, run_dir: Path, config: dict, resume_id: str | None):
        self.run = None
        if not enabled:
            return
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project, name=run_name, id=resume_id, resume='allow' if resume_id else None,
                                  config=config, dir=str(run_dir))
            wandb.define_metric('epoch')
            wandb.define_metric('global_step')
            wandb.define_metric('train/step_loss', step_metric='global_step')
            wandb.define_metric('train/grad_norm', step_metric='global_step')
            for prefix in ('train/', 'val/', 'proxy/', 'full/', 'time/', 'gpu/', 'lr', 'images/'):
                wandb.define_metric(prefix + '*' if prefix.endswith('/') else prefix, step_metric='epoch')
            (run_dir / 'wandb_id.txt').write_text(self.run.id)
            print(f'wandb: {self.run.url}', flush=True)
        except Exception as e:  # noqa: BLE001
            print(f'wandb отключён: {e!r}', flush=True)
            self.run = None

    def log(self, d: dict):
        if self.run:
            self.run.log(d)

    def images(self, panel: dict, df_val: pd.DataFrame, epoch: int):
        if not self.run or not panel:
            return
        imgs = []
        for idx in sorted(panel):
            img, gt, pred = panel[idx]
            r = df_val.iloc[idx]
            cap = f"{r['domain']} {'NEG ' + r['neg_pool'] if r['is_neg'] else 'pos area=' + str(round(float(r['gt_area']), 3))}"
            imgs.append(self.wandb.Image(img, caption=cap, masks={
                'ground_truth': {'mask_data': gt.astype(np.uint8), 'class_labels': {0: 'bg', 1: 'gt'}},
                'prediction': {'mask_data': pred.astype(np.uint8) * 2, 'class_labels': {0: 'bg', 2: 'pred'}}}))
        self.run.log({'images/proxy_panel': imgs, 'epoch': epoch})

    def summary(self, d: dict):
        if self.run:
            self.run.summary.update(d)

    def finish(self):
        if self.run:
            self.run.finish()


# ----------------------------------------------------------------------------- checkpoint
def rng_state() -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(s: dict) -> None:
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if s.get('cuda') is not None and torch.cuda.is_available():
        # число видимых GPU при resume может отличаться от сохранённого (4 → 2): восстанавливаем по доступным
        for i, st in enumerate(s['cuda'][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(st, i)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--version', default='v1')
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--img-size', default='256', help='сторона квадрата или WxH, кратно 32 (например 1024x768)')
    ap.add_argument('--decoder-channels', default=None, help='ширина UNet-декодера через запятую, например 128,64,32,16,8 (по умолчанию smp 256,...,16)')
    ap.add_argument('--val-limit', type=int, default=0, help='отладка: ограничить число val-tune строк (proxy-строки сохраняются)')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=32, help='на одну GPU')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--loss', default='bce', choices=['bce', 'bce_dice'])
    ap.add_argument('--mask-mix', type=float, default=0.0, help='mask-mixing по парам src/chng: вероятность на строку с оригиналом (0 — выключено); маски произвольной формы, площади — эмпирика ids9')
    ap.add_argument('--edge-weight', type=float, default=0.0, help='вес BCE на полосе вокруг контура GT (0 — выключено)')
    ap.add_argument('--init-weights', default=None, help='тёплый старт: загрузить только веса модели из чекпоинта (best_proxy.pth), оптимизатор/scheduler новые')
    ap.add_argument('--new-lr-mult', type=float, default=1.0, help='при --init-weights: параметры, которых нет в чекпоинте (новые модули), идут отдельной группой с lr × mult')
    ap.add_argument('--patch-head', action='store_true', help='patch-голова на глубоком признаке энкодера (forensic.py); proxy считается по слитой вероятности')
    ap.add_argument('--patch-weight', type=float, default=0.5, help='вес BCE patch-головы в общем loss')
    ap.add_argument('--patch-gamma', type=float, default=0.8, help='слияние p = max(p_pix, γ·up(p_patch)) на валидации/инференсе')
    ap.add_argument('--extra-input', default='none', choices=list(EXTRA_CHANNELS), help='доп. входные каналы (forensic.py): hp3 = SRM + медианный остаток + локальный std')
    ap.add_argument('--aux-head', action='store_true', help='image-level голова «есть ли правка» (smp aux_params), гейтинг при инференсе')
    ap.add_argument('--cls-weight', type=float, default=0.5, help='вес BCE image-level головы в общем loss')
    ap.add_argument('--aug', default='none', choices=['none', 'v1', 'v2'], help='аугментации робастности (data.AUG_PRESETS): v1 = rescale/JPEG q60–100/blur/noise (агрессивный), v2 = мягче, под распределение теста (JPEG q75–100, p 0.3)')
    ap.add_argument('--neg-src', type=int, default=0, help='сколько src-оригиналов train-сцен добавить в обучение как негативы (0 — не добавлять)')
    ap.add_argument('--scheduler', default='none', choices=['none', 'cosine'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8, help='DataLoader-воркеров на процесс')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--channels-last', action='store_true')
    ap.add_argument('--select-by', default='proxy', choices=['loss', 'proxy', 'full'])
    ap.add_argument('--full-every', type=int, default=1, help='считать full-метрику каждые N эпох; 0 — никогда')
    ap.add_argument('--patience', type=int, default=5, help='стоп, если критерий --select-by не улучшался N эпох подряд; 0 — без остановки (D-12)')
    ap.add_argument('--resume', default=None, help='auto = <run_dir>/last.pth, либо путь к last.pth')
    ap.add_argument('--rebuild-scheduler', action='store_true', help='при resume не восстанавливать scheduler, а построить заново на --epochs и прокрутить до текущей эпохи')
    ap.add_argument('--limit', type=int, default=0, help='отладка: ограничить число train-строк')
    ap.add_argument('--wandb', dest='wandb', action='store_true', default=True)
    ap.add_argument('--no-wandb', dest='wandb', action='store_false')
    ap.add_argument('--wandb-project', default='aic2026-detective')
    args = ap.parse_args()
    args.img_size = parse_size(args.img_size)          # (w, h)
    args.decoder_channels = parse_decoder(args.decoder_channels)

    rank, world, local_rank = ddp_setup()
    main_proc = rank == 0
    seed_everything(args.seed)
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    run_dir = runs_dir() / args.run_name
    args.split_file = split_path(args.version)
    if main_proc:
        run_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    # --- resume: читаем полный чекпоинт до создания модели
    ckpt = None
    if args.resume:
        rp = run_dir / 'last.pth' if args.resume == 'auto' else Path(args.resume)
        if rp.exists():
            ckpt = torch.load(rp, map_location='cpu', weights_only=False)
            if main_proc:
                print(f'resume from {rp}: epoch {ckpt["epoch"]} done', flush=True)
        elif main_proc:
            print(f'resume: {rp} не найден, начинаю с нуля', flush=True)
    start_epoch = (ckpt['epoch'] + 1) if ckpt else 1
    config = {**vars(args), 'split_file': str(args.split_file), 'world_size': world}
    if main_proc:
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str))

    df_train = load_split(args.version, 'train')
    if args.neg_src:
        src = df_train.loc[df_train['orgl_img_path'] != '', ['orgl_img_path', 'domain', 'group_id', 'fold', 'role', 'H', 'W']].drop_duplicates('orgl_img_path')
        src = src.sample(n=min(args.neg_src, len(src)), random_state=args.seed)
        neg = pd.DataFrame(dict(chng_img_path=src['orgl_img_path'].values, gt_path='', orgl_img_path='', domain=src['domain'].values,
                                generator='none', group_id=src['group_id'].values, fold=src['fold'].values, role='train', is_neg=1,
                                neg_pool='neg_src_train', in_proxy=0, H=src['H'].values, W=src['W'].values, gt_area=0.0))
        df_train = pd.concat([df_train, neg], ignore_index=True)
        if main_proc:
            print(f'neg-src: добавлено {len(neg)} оригиналов train-сцен как негативы; негативов в train теперь '
                  f'{int((df_train["is_neg"] == 1).sum())} / {len(df_train)}', flush=True)
    if args.limit:
        df_train = df_train.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)
    df_val = load_split(args.version, 'val-tune')
    if args.val_limit:
        keep = df_val[df_val['in_proxy'] == 1].index.tolist()[:args.val_limit]
        df_val = df_val.loc[keep].reset_index(drop=True)
    df_proxy = df_val[df_val['in_proxy'] == 1].reset_index(drop=True)
    # фиксированная панель картинок: 4 позитива, 2 neg_train, 2 neg_src из proxy
    rng = np.random.RandomState(args.seed)
    pv = df_val[df_val['in_proxy'] == 1]
    img_idx = set()
    for cond, k in ((pv['is_neg'] == 0, N_IMG_LOG // 2), (pv['neg_pool'] == 'neg_train', N_IMG_LOG // 4), (pv['neg_pool'] == 'neg_src', N_IMG_LOG // 4)):
        cand = pv.index[cond].values
        img_idx.update(int(i) for i in rng.choice(cand, size=min(k, len(cand)), replace=False))
    if main_proc:
        print(f'train={len(df_train)} val-tune={len(df_val)} proxy={len(df_proxy)} world={world} run_dir={run_dir}', flush=True)

    ds_train = SegDataset(df_train, args.img_size, train=True, aug=args.aug, mask_mix=args.mask_mix, extra_input=args.extra_input)
    sampler = DistributedSampler(ds_train, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                          num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_val = None
    if main_proc:
        dl_val = DataLoader(SegDataset(df_val, args.img_size, train=False, extra_input=args.extra_input), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    model = build_model(args.encoder, weights=None if (ckpt or args.init_weights) else 'imagenet', aux_head=args.aux_head, decoder_channels=args.decoder_channels,
                        extra_input=args.extra_input, patch_head=args.patch_head).to(device)
    new_param_names: set[str] = set()
    if args.init_weights and not ckpt:
        init = torch.load(args.init_weights, map_location='cpu', weights_only=False)
        res = model.load_state_dict(adapt_state_dict(init['model'], model), strict=False)
        if res.unexpected_keys:
            raise RuntimeError(f'init-weights: лишние ключи {res.unexpected_keys[:5]}')
        new_param_names = set(res.missing_keys)
        if main_proc:
            print(f'init-weights: {args.init_weights} (epoch {init.get("epoch")}), оптимизатор и scheduler новые; '
                  f'новых параметров (нет в чекпоинте): {len(new_param_names)} → lr × {args.new_lr_mult}: {sorted(new_param_names)[:6]}', flush=True)
    cls_criterion = nn.BCEWithLogitsLoss()
    patch_criterion = nn.BCEWithLogitsLoss()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    # find_unused_parameters: у timm-efficientnet часть параметров не попадает в граф (DDP падал с
    # «did not receive grad», indices 413–415); небольшой оверхед, зато работает для любого энкодера
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else model
    criterion = make_loss(args.loss)
    if new_param_names and args.new_lr_mult != 1.0:
        base = [p for n, p in model.named_parameters() if n not in new_param_names]
        new = [p for n, p in model.named_parameters() if n in new_param_names]
        optimizer = torch.optim.Adam([{'params': base, 'lr': args.lr}, {'params': new, 'lr': args.lr * args.new_lr_mult}])
    else:
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.scheduler == 'cosine' else None
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    best = {'loss': float('inf'), 'proxy': -1.0, 'full': -1.0}
    best_epoch = {'loss': 0, 'proxy': 0, 'full': 0}
    global_step = 0
    wandb_id = None
    if ckpt:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and ckpt.get('scheduler'):
            if ckpt.get('args', {}).get('epochs') == args.epochs and not args.rebuild_scheduler:
                scheduler.load_state_dict(ckpt['scheduler'])
            else:
                # Горизонт изменён (например, 20 → 30) или --rebuild-scheduler: строим cosine на новом горизонте заново
                # и прокручиваем его на ckpt['epoch'] шагов от базового lr. Через last_epoch=N делать НЕЛЬЗЯ:
                # рекурсивная формула CosineAnnealingLR отталкивается от текущего lr, а он после старого
                # расписания уже 0 — и остаётся 0 навсегда.
                for g in optimizer.param_groups:
                    g['lr'] = args.lr
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                for _ in range(ckpt['epoch']):
                    scheduler.step()
                if main_proc:
                    print(f'resume: epochs {ckpt.get("args", {}).get("epochs")} → {args.epochs}, scheduler пересоздан, '
                          f'lr на следующей эпохе {optimizer.param_groups[0]["lr"]:.2e}', flush=True)
        scaler.load_state_dict(ckpt['scaler'])
        best, best_epoch, global_step = ckpt['best'], ckpt['best_epoch'], ckpt.get('global_step', 0)
        wandb_id = ckpt.get('wandb_id')
        if main_proc and ckpt.get('rng'):
            set_rng_state(ckpt['rng'])

    wb = WandbLogger(args.wandb and main_proc, args.wandb_project, args.run_name, run_dir, config, wandb_id) if main_proc else None
    log_f = open(run_dir / 'log.jsonl', 'a', encoding='utf-8') if main_proc else None
    steps_f = open(run_dir / 'steps.jsonl', 'a', encoding='utf-8') if main_proc else None
    stop_flag = torch.zeros(1, device=device)
    params = [p for p in ddp_model.parameters() if p.requires_grad]

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        ddp_model.train()
        tr_loss, nb, gn_sum, gn_n = 0.0, 0, 0.0, 0
        torch.cuda.reset_peak_memory_stats(device)
        for batch in tqdm(dl_train, desc=f'train ep{epoch}', leave=False, disable=None if main_proc else True):
            x = batch['image'].to(device, non_blocking=True)
            y = batch['mask'].to(device, non_blocking=True)
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=args.amp):
                logits, img_logit, patch_logit = forward(ddp_model, x)
            loss = criterion(logits.float(), y)
            if args.edge_weight > 0:
                loss = loss + args.edge_weight * edge_loss(logits.float(), y)
            if img_logit is not None:
                loss = loss + args.cls_weight * cls_criterion(img_logit.float(), batch['label'].to(device))
            if patch_logit is not None:
                loss = loss + args.patch_weight * patch_criterion(patch_logit.float(), patch_targets(y, patch_logit.shape[-2:]))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(params, max_norm=1e9).item()   # только измеряем, не режем
            scaler.step(optimizer)
            scaler.update()
            li = loss.item(); tr_loss += li; nb += 1; global_step += 1
            if np.isfinite(gn):
                gn_sum += gn; gn_n += 1
            if main_proc and global_step % STEP_LOG_EVERY == 0:
                if wb:
                    wb.log({'train/step_loss': li, 'train/grad_norm': gn, 'global_step': global_step})
                steps_f.write(json.dumps({'global_step': global_step, 'epoch': epoch, 'loss': li, 'grad_norm': gn,
                                          'lr': optimizer.param_groups[0]['lr']}) + chr(10)); steps_f.flush()
        if scheduler:
            scheduler.step()
        tr_loss = all_reduce_mean(tr_loss / max(nb, 1), device)
        t_train = time.time() - t0

        if main_proc:
            va_loss, pm, panel = validate(model, dl_val, criterion, device, df_val, args.amp, img_idx,
                                          patch_gamma=args.patch_gamma if args.patch_head else 0.0)
            fm = None
            if args.full_every and not args.val_limit and (epoch % args.full_every == 0 or epoch == args.epochs):
                fm = full_metric(model, df_proxy, args, device, run_dir, epoch)
            lr_now = optimizer.param_groups[0]['lr']
            rec = dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss, proxy=pm, full=fm, lr=lr_now,
                       grad_norm=gn_sum / max(gn_n, 1), time_s=round(time.time() - t0), train_time_s=round(t_train),
                       gpu_max_mem_gb=round(torch.cuda.max_memory_allocated(device) / 2**30, 2))
            log_f.write(json.dumps(rec, ensure_ascii=False) + '\n'); log_f.flush()
            print(f'ep{epoch} train={tr_loss:.4f} val={va_loss:.4f} | proxy dice={pm["dice_pos"]:.4f} '
                  f'fpr={pm["fpr_main"]:.4f} aic={pm["aic"]:.4f}'
                  + (f' (ungated {pm["aic_ungated"]:.4f}; cls acc={pm["cls"]["acc"]:.3f} tpr={pm["cls"]["tpr"]:.3f} tnr={pm["cls"]["tnr"]:.3f})' if 'cls' in pm else '')
                  + (f' [без patch-головы aic={pm["nopatch"]["aic"]:.4f} dice={pm["nopatch"]["dice_pos"]:.4f}]' if 'nopatch' in pm else '')
                  + (f' | full dice={fm["dice_pos"]:.4f} fpr={fm["fpr_main"]:.4f} aic={fm["aic"]:.4f}' if fm else '')
                  + f' | lr={lr_now:.2e} gn={rec["grad_norm"]:.2f} mem={rec["gpu_max_mem_gb"]}G | {rec["time_s"]}s (train {rec["train_time_s"]}s)', flush=True)

            if va_loss < best['loss']:
                best['loss'] = va_loss; best_epoch['loss'] = epoch
            if pm['aic'] > best['proxy']:
                best['proxy'] = pm['aic']; best_epoch['proxy'] = epoch
            if fm and fm['aic'] > best['full']:
                best['full'] = fm['aic']; best_epoch['full'] = epoch
            weights = dict(epoch=epoch, model=model.state_dict(), encoder=args.encoder, img_size=list(args.img_size), aux_head=args.aux_head,
                           decoder_channels=list(args.decoder_channels) if args.decoder_channels else None, metrics=rec,
                           extra_input=args.extra_input, patch_head=args.patch_head, patch_gamma=args.patch_gamma if args.patch_head else 0.0)
            for crit in ('loss', 'proxy', 'full'):
                if best_epoch[crit] == epoch:
                    torch.save(weights, run_dir / f'best_{crit}.pth')
            full_state = dict(weights, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
                              scaler=scaler.state_dict(), best=best, best_epoch=best_epoch, global_step=global_step,
                              rng=rng_state(), wandb_id=wb.run.id if (wb and wb.run) else None, args=vars(args))
            torch.save(full_state, run_dir / 'last.pth.tmp'); os.replace(run_dir / 'last.pth.tmp', run_dir / 'last.pth')

            save_panel(panel, df_val, run_dir, epoch)
            if wb:
                d = {'epoch': epoch, 'train/loss': tr_loss, 'val/loss': va_loss, 'val/gap_val_minus_train': va_loss - tr_loss,
                     'lr': lr_now, 'train/grad_norm_epoch': rec['grad_norm'], 'time/epoch_s': rec['time_s'],
                     'time/train_s': rec['train_time_s'], 'gpu/max_mem_gb': rec['gpu_max_mem_gb'],
                     'proxy/dice_pos': pm['dice_pos'], 'proxy/fpr_main': pm['fpr_main'], 'proxy/aic': pm['aic'],
                     'proxy/empty_pred_on_pos': pm['empty_pred_on_pos'], 'proxy/neg_pred_area_median': pm['neg_pred_area_median'],
                     'best/proxy_aic': best['proxy'], 'best/epoch_proxy': best_epoch['proxy'],
                     'epochs_since_best': epoch - best_epoch[args.select_by]}
                d.update({f'proxy/fpr_{k}': v for k, v in pm['fpr'].items()})
                d.update({f'proxy/dice_domain_{k}': v for k, v in pm['dice_by_domain'].items()})
                d.update({f'proxy/dice_area_{k}': v for k, v in pm['dice_by_area'].items()})
                if 'cls' in pm:
                    d.update({'proxy/aic_ungated': pm['aic_ungated'], 'proxy/dice_pos_ungated': pm['dice_pos_ungated'],
                              'proxy/fpr_main_ungated': pm['fpr_main_ungated'], 'cls/acc': pm['cls']['acc'], 'cls/tpr': pm['cls']['tpr'], 'cls/tnr': pm['cls']['tnr']})
                if 'nopatch' in pm:
                    d.update({'proxy/aic_nopatch': pm['nopatch']['aic'], 'proxy/dice_pos_nopatch': pm['nopatch']['dice_pos'], 'proxy/fpr_main_nopatch': pm['nopatch']['fpr_main']})
                if fm:
                    d.update({'full/dice_pos': fm['dice_pos'], 'full/fpr_main': fm['fpr_main'], 'full/aic': fm['aic']})
                    d.update({f'full/fpr_{k}': v for k, v in fm['fpr'].items()})
                wb.log(d)
                wb.images(panel, df_val, epoch)
                wb.summary({'best': best, 'best_epoch': best_epoch})

            if args.patience and epoch - best_epoch[args.select_by] >= args.patience:
                print(f'early stop: {args.select_by} не улучшался {args.patience} эпох (лучшая эпоха {best_epoch[args.select_by]})', flush=True)
                stop_flag.fill_(1)
        if is_dist():
            dist.broadcast(stop_flag, src=0)
        barrier()
        if stop_flag.item():
            break

    if main_proc:
        log_f.close(); steps_f.close()
        src = run_dir / f'best_{args.select_by}.pth'
        if src.exists():
            shutil.copyfile(src, run_dir / 'best.pth')
        (run_dir / 'best.json').write_text(json.dumps({'select_by': args.select_by, 'best': best, 'best_epoch': best_epoch}, indent=2))
        print('done; best:', best, best_epoch, flush=True)
        if wb:
            wb.finish()
    if is_dist():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
