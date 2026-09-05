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

cv2.setNumThreads(0)

AUG_PRESETS = {
    'none': None,
    'v1': dict(p_rescale=0.3, scale=(0.5, 1.0), p_jpeg=0.5, q=(60, 100), p_blur=0.15, sigma=(0.3, 1.0), p_noise=0.15, noise=(1.0, 5.0)),
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


class SegDataset(Dataset):
    """rows: DataFrame split-а (chng_img_path, gt_path, is_neg, neg_pool, in_proxy, H, W).
    Для строк без gt_path (neg_src / neg_nb / тест) маска нулевая."""

    def __init__(self, df, img_size, train: bool, root=None, path_col: str = 'chng_img_path', aug: str | None = None):
        self.df = df.reset_index(drop=True)
        self.img_size = parse_size(img_size)   # (w, h)
        self.train = train
        self.root = root or data_dir()
        self.path_col = path_col
        self.aug = AUG_PRESETS[aug or 'none'] if train else None
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
        if self.aug:
            img = augment_postprocess(img, self._get_rng(), self.aug)
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        gt_path = r.get('gt_path', '')
        if gt_path:
            m = load_gt(self.root / gt_path).astype(np.uint8)
            m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
        else:
            m = np.zeros((sh, sw), np.uint8)
        if self.train and self._get_rng().random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            m = np.ascontiguousarray(m[:, ::-1])
        is_neg = int(r.get('is_neg', 1 if not gt_path else 0))
        return {
            'image': torch.from_numpy(normalize(img)),
            'mask': torch.from_numpy(m.astype(np.float32)).unsqueeze(0),
            'label': torch.tensor([0.0 if is_neg else 1.0]),   # image-level: есть ли манипуляция
            'idx': idx,
            'orig_h': h,
            'orig_w': w,
        }
