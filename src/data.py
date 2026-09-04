"""data.py — Dataset для обучения/валидации/инференса.

Картинка ресайзится к квадрату img_size (как в бейзлайне организаторов), маска — nearest.
Аугментации пока минимальные (hflip); forgery-специфичные (JPEG, blur, noise, resize) —
следующий шаг, добавлять сюда через параметр `aug`.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import data_dir, load_gt, load_rgb, normalize, parse_size

cv2.setNumThreads(0)


class SegDataset(Dataset):
    """rows: DataFrame split-а (chng_img_path, gt_path, is_neg, neg_pool, in_proxy, H, W).
    Для строк без gt_path (neg_src / neg_nb / тест) маска нулевая."""

    def __init__(self, df, img_size, train: bool, root=None, path_col: str = 'chng_img_path'):
        self.df = df.reset_index(drop=True)
        self.img_size = parse_size(img_size)   # (w, h)
        self.train = train
        self.root = root or data_dir()
        self.path_col = path_col
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = load_rgb(self.root / r[self.path_col])
        h, w = img.shape[:2]
        sw, sh = self.img_size
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        gt_path = r.get('gt_path', '')
        if gt_path:
            m = load_gt(self.root / gt_path).astype(np.uint8)
            m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
        else:
            m = np.zeros((sh, sw), np.uint8)
        if self.train and self.rng.random() < 0.5:
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
