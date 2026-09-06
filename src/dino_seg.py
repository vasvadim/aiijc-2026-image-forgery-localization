"""dino_seg.py — сегментатор на ViT-энкодере DINOv3/DINOv2 (timm) с лёгким FPN-декодером.

Зачем: RESEARCH.md §2/§5 — DINO-признаки сильнее ImageNet-бэкбонов для локализации правок, но ViT
влезает в 100 GFLOPs только на 640–768 px. Энкодер отдаёт признаки в 1/16 (патч 16), берём 4
промежуточных блока, проецируем в `ch` каналов, суммируем, поднимаем до 1/4 двумя conv-блоками и
дальше билинейно до входа. Image-level голова — линейный слой на среднем токене последнего блока
(аналог aux_params smp). Имя энкодера в train.py: `dino:<timm-имя>`, например
`dino:vit_small_patch16_dinov3.lvd1689m`.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DinoSeg(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, aux_head: bool = True, ch: int = 128, indices=(3, 6, 9, 11)):
        super().__init__()
        self.enc = timm.create_model(name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        C = self.enc.embed_dim
        self.indices = list(indices)
        self.proj = nn.ModuleList([nn.Conv2d(C, ch, 1) for _ in self.indices])
        self.fuse = ConvBNAct(ch, ch)          # 1/16
        self.up1 = ConvBNAct(ch, ch)           # 1/8
        self.up2 = ConvBNAct(ch, ch // 2)      # 1/4
        self.head = nn.Conv2d(ch // 2, 1, 1)
        self.aux = nn.Sequential(nn.Dropout(0.2), nn.Linear(C, 1)) if aux_head else None

    def forward(self, x):
        feats = self.enc.forward_intermediates(x, indices=self.indices, output_fmt='NCHW', intermediates_only=True)
        f = self.proj[0](feats[0])
        for p, t in zip(self.proj[1:], feats[1:]):
            f = f + p(t)
        f = self.fuse(f)
        f = self.up1(F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False))
        f = self.up2(F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False))
        logits = F.interpolate(self.head(f), size=x.shape[-2:], mode='bilinear', align_corners=False)
        if self.aux is None:
            return logits
        img_logit = self.aux(feats[-1].mean(dim=(2, 3)))
        return logits, img_logit
