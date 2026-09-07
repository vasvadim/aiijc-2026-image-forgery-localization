"""data.py — Dataset для обучения/валидации/инференса.

Картинка ресайзится к (w, h) img_size, маска — nearest. Аугментации (train=True): hflip всегда;
`aug='v1'` — робастность к пост-обработке (RESEARCH.md §3.5: модель, обученная на q100, теряет
0.2–0.5 Dice при q85): на исходном разрешении, до ресайза к img_size, случайно применяются
rescale вниз-вверх (p=0.3, масштаб 0.5–1.0), JPEG-пересжатие (p=0.5, q∈[60,100]), gaussian blur
(p=0.15, σ 0.3–1.0), gaussian noise (p=0.15, σ 1–5). Маска не трогается (геометрия не меняется).
Негативы аугментируются так же, чтобы «сжато» не стало признаком правки.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import data_dir, load_gt, load_rgb, normalize, parse_size
from forensic import extra_channels

cv2.setNumThreads(0)

AUG_PRESETS = {
    'none': None,
    'v1': dict(p_rescale=0.3, scale=(0.5, 1.0), p_jpeg=0.5, q=(60, 100), p_blur=0.15, sigma=(0.3, 1.0), p_noise=0.15, noise=(1.0, 5.0)),
    # v2 — под распределение теста (RESEARCH.md §3.2: 64 % q100, 13 % q95, 23 % q80–94; q<80 нет):
    # ~55 % картинок остаются нетронутыми, диапазон q75–100
    'v2': dict(p_rescale=0.2, scale=(0.5, 1.0), p_jpeg=0.3, q=(75, 100), p_blur=0.1, sigma=(0.3, 1.0), p_noise=0.1, noise=(1.0, 5.0)),
}
_INTERP = (cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC)


def augment_postprocess(img: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """uint8 HxWx3 → uint8 HxWx3 того же размера. Порядок: rescale → JPEG → blur → noise."""
    h, w = img.shape[:2]
    if rng.random() < cfg['p_rescale']:
        s = rng.uniform(*cfg['scale'])
        nw, nh = max(32, int(w * s)), max(32, int(h * s))
        img = cv2.resize(img, (nw, nh), interpolation=_INTERP[rng.integers(3)])
        img = cv2.resize(img, (w, h), interpolation=_INTERP[rng.integers(3)])
    if rng.random() < cfg['p_jpeg']:
        q = int(rng.integers(cfg['q'][0], cfg['q'][1] + 1))
        ok, buf = cv2.imencode('.jpg', img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    if rng.random() < cfg['p_blur']:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(*cfg['sigma'])))
    if rng.random() < cfg['p_noise']:
        noise = rng.normal(0, rng.uniform(*cfg['noise']), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


MIX_DEFAULT = dict(areas='ids9', feather=0.0, shapes=('box', 'ellipse'))
# v2 (07.09, RESEARCH §9 / обсуждение): areas='small' — лог-равномерно 0.3–5 % кадра (дефицит Dice — маски 1–5 %),
# 'mix' — 60 % small + 40 % эмпирика ids9; feather — растушёвка края вставки (σ ~ U(0, feather) px), как в
# self-blended методах (SBI/PIDA), чтобы сеть не училась на жёсткую границу склейки; shapes — box/ellipse/poly.
SMALL_AREA = (0.003, 0.05)


def _region(h, w, cy, cx, hw, hh, shape, poly=None):
    """Булева маска региона R с центром (cy, cx) и полуразмерами (hh, hw): 'box', 'ellipse' или 'poly'
    (звёздчатый многоугольник, poly = (углы, радиусы 0.5–1) в единичных координатах)."""
    R = np.zeros((h, w), bool)
    if shape == 'poly':
        pts = np.stack([cx + hw * poly[1] * np.cos(poly[0]), cy + hh * poly[1] * np.sin(poly[0])], 1)
        cv2.fillPoly(R.view(np.uint8), [np.round(pts).astype(np.int32)], 1)
        return R
    y0, y1 = max(0, int(cy - hh)), min(h, int(cy + hh) + 1)
    x0, x1 = max(0, int(cx - hw)), min(w, int(cx + hw) + 1)
    if y1 <= y0 or x1 <= x0:
        return R
    if shape == 'ellipse':
        yy, xx = np.ogrid[y0:y1, x0:x1]
        R[y0:y1, x0:x1] = ((yy - cy) / max(hh, 1)) ** 2 + ((xx - cx) / max(hw, 1)) ** 2 <= 1.0
    else:
        R[y0:y1, x0:x1] = True
    return R


def _sample_area(rng, target_areas, mode: str) -> float:
    if mode == 'small' or (mode == 'mix' and rng.random() < 0.6):
        return float(np.exp(rng.uniform(np.log(SMALL_AREA[0]), np.log(SMALL_AREA[1]))))
    return float(rng.choice(target_areas))


def mask_mix(chng: np.ndarray, src: np.ndarray, gt: np.ndarray, rng, target_areas, cfg: dict | None = None):
    """Новый позитив из выровненной пары (src, chng, GT): режим crop (p=0.7) — в src вставляем кусок chng
    внутри региона R, маска = R ∩ GT, размер R подбирается под целевую площадь (cfg['areas']: 'ids9' — эмпирика
    target_areas, 'small' — 0.3–5 % кадра, 'mix' — 60/40); режим hole — в chng вставляем src внутри R, маска = GT \\ R.
    R — box / ellipse / poly (cfg['shapes']) с центром в случайной точке GT; край вставки растушёван
    (cfg['feather'] px). Возвращает (img, mask) или None."""
    cfg = {**MIX_DEFAULT, **(cfg or {})}
    h, w = gt.shape
    ys, xs = np.nonzero(gt)
    if len(ys) < 64:
        return None
    k = rng.integers(len(ys)); cy, cx = int(ys[k]), int(xs[k])
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bh, bw = max(8, (y1 - y0 + 1) / 2), max(8, (x1 - x0 + 1) / 2)
    shape = cfg['shapes'][rng.integers(len(cfg['shapes']))]
    poly = None
    if shape == 'poly':
        n = int(rng.integers(6, 13))
        ang = np.sort(rng.uniform(0, 2 * np.pi, n)); rad = rng.uniform(0.5, 1.0, n)
        poly = (ang, rad)
    gt_area = int(gt.sum())
    if rng.random() < 0.7:   # crop: подобрать масштаб R под целевую площадь (бисекция по scale)
        target = _sample_area(rng, target_areas, cfg['areas']) * h * w
        target = min(target, 0.9 * gt_area)
        if target < 0.002 * h * w:
            return None
        lo, hi, best = 0.02, 2.0, None
        for _ in range(8):
            s = (lo + hi) / 2
            R = _region(h, w, cy, cx, s * bw, s * bh, shape, poly)
            a = int((R & gt).sum())
            best = (R, a)
            if a < target:
                lo = s
            else:
                hi = s
        R, a = best
        new_mask = R & gt
        if a < 0.002 * h * w or a > 0.95 * gt_area:
            return None
        sigma = float(rng.uniform(0, cfg['feather'])) if cfg['feather'] > 0 else 0.0
        if sigma > 0.3:
            alpha = cv2.GaussianBlur(R.astype(np.float32), (0, 0), sigma)[..., None]
            out = np.clip(src.astype(np.float32) * (1 - alpha) + chng.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        else:
            out = src.copy(); out[R] = chng[R]
    else:                    # hole: вырезать из GT кусок, вернув туда оригинал
        s = rng.uniform(0.2, 0.6)
        R = _region(h, w, cy, cx, s * bw, s * bh, shape, poly)
        new_mask = gt & ~R
        if new_mask.sum() < max(64, 0.05 * gt_area) or (gt & R).sum() < 0.05 * gt_area:
            return None
        out = chng.copy(); out[R] = src[R]
    return out, new_mask.astype(np.uint8)


class SegDataset(Dataset):
    """rows: DataFrame split-а (chng_img_path, gt_path, is_neg, neg_pool, in_proxy, H, W).
    Для строк без gt_path (neg_src / neg_nb / тест) маска нулевая."""

    def __init__(self, df, img_size, train: bool, root=None, path_col: str = 'chng_img_path', aug: str | None = None,
                 mask_mix: float = 0.0, extra_input: str = 'none', mix_cfg: dict | None = None):
        self.df = df.reset_index(drop=True)
        self.mix_cfg = {**MIX_DEFAULT, **(mix_cfg or {})}
        self.extra_input = extra_input   # доп. входные каналы (forensic.extra_channels), считаются после ресайза/flip
        self.img_size = parse_size(img_size)   # (w, h)
        self.train = train
        self.root = root or data_dir()
        self.path_col = path_col
        self.aug = AUG_PRESETS[aug or 'none'] if train else None
        # mask-mixing: вероятность на строку с оригиналом; целевые площади — эмпирика ids9 (домен без src, где болит)
        self.mask_mix = float(mask_mix) if train else 0.0
        ta = self.df.loc[(self.df.get('domain') == 'ids9') & (self.df.get('is_neg', 0) == 0), 'gt_area'] if 'gt_area' in self.df else None
        self.target_areas = ta.dropna().values.astype(np.float32) if ta is not None and len(ta) > 100 else np.array([0.01, 0.02, 0.05, 0.1, 0.2], np.float32)
        self.mix_stats = dict(tried=0, done=0)
        self.rng = np.random.default_rng()
        self._rng_pid = os.getpid()

    def __len__(self):
        return len(self.df)

    def _get_rng(self):
        # DataLoader-воркеры форкаются с копией rng — переинициализируем в каждом процессе
        if os.getpid() != self._rng_pid:
            self.rng = np.random.default_rng([os.getpid(), int(torch.initial_seed()) % (2**31)])
            self._rng_pid = os.getpid()
        return self.rng

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = load_rgb(self.root / r[self.path_col])
        h, w = img.shape[:2]
        sw, sh = self.img_size
        gt_path = r.get('gt_path', '')
        m_full = None
        src_path = r.get('orgl_img_path', '')
        if self.mask_mix > 0 and gt_path and isinstance(src_path, str) and src_path and self._get_rng().random() < self.mask_mix:
            self.mix_stats['tried'] += 1
            src = load_rgb(self.root / src_path)
            if src.shape == img.shape:
                gt_full = load_gt(self.root / gt_path)
                mixed = mask_mix(img, src, gt_full, self._get_rng(), self.target_areas, self.mix_cfg)
                if mixed is not None:
                    img, m_full = mixed
                    self.mix_stats['done'] += 1
        if self.aug:
            img = augment_postprocess(img, self._get_rng(), self.aug)
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        if m_full is not None:
            m = cv2.resize(m_full, (sw, sh), interpolation=cv2.INTER_NEAREST)
        elif gt_path:
            m = load_gt(self.root / gt_path).astype(np.uint8)
            m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
        else:
            m = np.zeros((sh, sw), np.uint8)
        if self.train and self._get_rng().random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            m = np.ascontiguousarray(m[:, ::-1])
        is_neg = int(r.get('is_neg', 1 if not gt_path else 0))
        x = normalize(img)
        if self.extra_input != 'none':
            x = np.concatenate([x, extra_channels(img, self.extra_input)], axis=0)
        return {
            'image': torch.from_numpy(x),
            'mask': torch.from_numpy(m.astype(np.float32)).unsqueeze(0),
            'label': torch.tensor([0.0 if is_neg else 1.0]),   # image-level: есть ли манипуляция
            'idx': idx,
            'orig_h': h,
            'orig_w': w,
        }
