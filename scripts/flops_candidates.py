# -*- coding: utf-8 -*-
"""FLOPs кандидатов (CPU, FlopCounterMode, 1 MAC = 2 FLOPs), веса не грузим."""
import sys, json, torch, timm
import segmentation_models_pytorch as smp
from torch.utils.flop_counter import FlopCounterMode
torch.set_num_threads(16)
res = []
def flops(model, w, h):
    model.eval(); x = torch.randn(1, 3, h, w)
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc: model(x)
    return fc.get_total_flops() / 1e9
def unet(enc, dec=None):
    kw = dict(decoder_channels=dec) if dec else {}
    return smp.Unet(enc, encoder_weights=None, classes=1, aux_params=dict(classes=1, pooling='avg', dropout=0.2), **kw)
D128 = (128, 64, 32, 16, 8)
cands = [
 ('Unet timm-efficientnet-b4 dec128', lambda: unet('timm-efficientnet-b4', D128), [(1024,1024)]),
 ('Unet tu-convnext_tiny dec128', lambda: unet('tu-convnext_tiny', D128), [(1024,1024),(1024,768),(768,768),(896,896)]),
 ('Unet tu-convnext_small dec128', lambda: unet('tu-convnext_small', D128), [(768,768),(896,896),(1024,768)]),
 ('Unet tu-convnext_nano dec128', lambda: unet('tu-convnext_nano', D128), [(1024,1024)]),
 ('Segformer mit_b2', lambda: smp.Segformer('mit_b2', encoder_weights=None, classes=1, aux_params=dict(classes=1, pooling='avg', dropout=0.2)), [(1024,1024),(768,768)]),
 ('Segformer mit_b3', lambda: smp.Segformer('mit_b3', encoder_weights=None, classes=1, aux_params=dict(classes=1, pooling='avg', dropout=0.2)), [(1024,1024),(768,768)]),
 ('Segformer tu-convnext_tiny', lambda: smp.Segformer('tu-convnext_tiny', encoder_weights=None, classes=1), [(1024,1024)]),
 ('Unet mit_b2 dec128', lambda: unet('mit_b2', D128), [(1024,1024)]),
 ('Unet tu-efficientnetv2_s dec128', lambda: unet('tu-efficientnetv2_s', D128), [(1024,1024)]),
 ('Unet tu-efficientnet_b5 dec128', lambda: unet('tu-efficientnet_b5', D128), [(1024,1024),(896,896)]),
 ('ViT-S/16 dinov3 encoder only', lambda: timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=False, img_size=(SZ,SZ)), [(512,512),(768,768),(1024,1024)]),
 ('ViT-S+/16 dinov3 encoder only', lambda: timm.create_model('vit_small_plus_patch16_dinov3.lvd1689m', pretrained=False, img_size=(SZ,SZ)), [(512,512),(768,768)]),
 ('ViT-B/16 dinov3 encoder only', lambda: timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=False, img_size=(SZ,SZ)), [(512,512),(768,768)]),
 ('ViT-B/14 dinov2 encoder only', lambda: timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, img_size=(SZ,SZ)), [(504,504)]),
]
for name, mk, sizes in cands:
    for (w, h) in sizes:
        SZ = w
        try:
            m = mk(); g = flops(m, w, h)
            n = sum(p.numel() for p in m.parameters()) / 1e6
            print(f'{name:38s} {w}x{h}: {g:7.1f} GFLOPs  {n:6.1f}M params', flush=True); res.append(dict(name=name, w=w, h=h, gflops=g, params_m=n))
        except Exception as e:
            print(f'{name:38s} {w}x{h}: ERR {type(e).__name__}: {str(e)[:120]}', flush=True)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
