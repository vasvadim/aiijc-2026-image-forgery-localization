"""forensic.py — дешёвые надстройки над smp.Unet (RESEARCH.md §8.3, 07.09), обе ≈ 0 FLOPs:

  • patch-голова (`--patch-head`): Conv1×1 на самом глубоком признаке энкодера (stride 32 → 32×32 при 1024²) —
    «в патче есть правка». Обучение: BCE с мягкой целью clip(доля GT в патче / 0.1, 0, 1), вес --patch-weight.
    Инференс: p = max(p_pix, γ·up(p_patch)), γ = --patch-gamma (слияние по max — как у победителей
    MediaEval-25 SynthIM и Loupe). Смысл: маленькая правка (1–5 % кадра) — это целый положительный патч, а не
    3 % пикселей, поэтому голова видит то, что пиксельный декодер «не дотягивает» до порога.

  • extra-вход (`--extra-input hp3`): 3 доп. канала к RGB — SRM 5×5 (KB) остаток, медианный остаток (k=3),
    локальный std остатка (окно 16). Считаются в Dataset после ресайза к рабочему размеру (одинаково в train и
    на инференсе). В энкодере отдельная свёртка stem_extra (zero-init) прибавляется к выходу conv_stem, так что
    веса контроля не трогаются, а новые параметры идут отдельной группой с повышенным lr (--new-lr-mult).
"""
from __future__ import annotations

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

SRM_KB = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12
EXTRA_CHANNELS = {'none': 0, 'hp3': 3}
PATCH_SOFT_AREA = 0.1   # доля GT в патче, при которой цель = 1


def extra_channels(img_uint8: np.ndarray, kind: str) -> np.ndarray | None:
    """HxWx3 uint8 (в рабочем разрешении) → (k,H,W) float32 доп. каналов; None для kind='none'."""
    if kind == 'none':
        return None
    if kind != 'hp3':
        raise ValueError(kind)
    g8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    g = g8.astype(np.float32)
    srm = cv2.filter2D(g, -1, SRM_KB)
    med = g - cv2.medianBlur(g8, 3).astype(np.float32)
    m1 = cv2.blur(med, (16, 16)); m2 = cv2.blur(med * med, (16, 16))
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    out = np.stack([np.clip(srm / 8, -4, 4), np.clip(med / 8, -4, 4), np.clip(std / 8, 0, 4)]).astype(np.float32)
    return np.ascontiguousarray(out)


class StemPlus(nn.Module):
    """conv_stem(RGB) + stem_extra(доп. каналы); stem_extra zero-init → на старте сеть = контроль."""

    def __init__(self, orig: nn.Conv2d, extra_in: int):
        super().__init__()
        self.orig = orig
        self.extra = nn.Conv2d(extra_in, orig.out_channels, orig.kernel_size, orig.stride, orig.padding, bias=False)
        nn.init.zeros_(self.extra.weight)

    def forward(self, x):
        n = self.orig.in_channels
        return self.orig(x[:, :n]) + self.extra(x[:, n:])


class ForensicUnet(smp.Unet):
    """smp.Unet + опциональные StemPlus и patch-голова. forward → (mask_logits, img_logit | None, patch_logits | None)."""

    def __init__(self, *args, extra_in: int = 0, patch_head: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_in = extra_in
        if extra_in:
            self.encoder.conv_stem = StemPlus(self.encoder.conv_stem, extra_in)
        self.patch_head = nn.Conv2d(self.encoder.out_channels[-1], 1, kernel_size=1) if patch_head else None

    def forward(self, x):
        feats = self.encoder(x)
        masks = self.segmentation_head(self.decoder(feats))
        img = self.classification_head(feats[-1]) if self.classification_head is not None else None
        patch = self.patch_head(feats[-1]) if self.patch_head is not None else None
        return masks, img, patch


def adapt_state_dict(sd: dict, model: nn.Module) -> dict:
    """Веса обычного smp.Unet (контроль) → ключи ForensicUnet: conv_stem.weight → conv_stem.orig.weight."""
    if isinstance(getattr(model.encoder, 'conv_stem', None), StemPlus) and 'encoder.conv_stem.weight' in sd:
        sd = dict(sd); sd['encoder.conv_stem.orig.weight'] = sd.pop('encoder.conv_stem.weight')
    return sd


def patch_targets(y: torch.Tensor, size) -> torch.Tensor:
    """(B,1,H,W) маска → (B,1,h,w) мягкая цель патч-головы: clip(доля GT в патче / PATCH_SOFT_AREA, 0, 1)."""
    frac = F.adaptive_avg_pool2d(y, size)
    return torch.clamp(frac / PATCH_SOFT_AREA, 0, 1)


def fuse_patch(prob: torch.Tensor, patch_logit: torch.Tensor | None, gamma: float) -> torch.Tensor:
    """p = max(p_pix, γ·up(p_patch)); при γ ≤ 0 или без головы — p_pix."""
    if patch_logit is None or gamma <= 0:
        return prob
    pp = torch.sigmoid(patch_logit.float())
    pp = F.interpolate(pp, size=prob.shape[-2:], mode='bilinear', align_corners=False)
    return torch.maximum(prob, gamma * pp)
