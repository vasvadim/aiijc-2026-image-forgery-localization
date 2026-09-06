#!/usr/bin/env python
"""measure.py — строгие FLOPs (1 MAC = 2 FLOPs, torch.utils.flop_counter) и latency чекпоинта.

Лимиты задачи: ≤ 100 GFLOPs и ≤ 50 ms на изображение (HACKATHON.md §5). FlopCounterMode —
эталонный счётчик организаторов. Latency меряется на batch=1 в рабочем разрешении, с прогревом,
и это только forward: препроцессинг/постпроцессинг сюда не входят (открытый вопрос §10).

    python src/measure.py --checkpoint runs/.../best.pth [--img-size 256] [--iters 100]
    python src/measure.py --encoder resnet34 --img-size 512   # без чекпоинта, случайные веса
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))


def measure(model, img_size, iters: int, device, in_ch: int = 3) -> dict:
    from common import parse_size
    w, h = parse_size(img_size)
    model.eval()
    x = torch.randn(1, in_ch, h, w, device=device)
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            model(x)
        flops = fc.get_total_flops()
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000
    return dict(img_size=[w, h], gflops=flops / 1e9, latency_ms=ms, device=torch.cuda.get_device_name(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint')
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--img-size', default=None, help='WxH или сторона')
    ap.add_argument('--decoder-channels', default=None)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--out')
    args = ap.parse_args()
    device = torch.device('cuda')
    in_ch = 3
    if args.checkpoint:
        from forensic import EXTRA_CHANNELS
        from predict import load_model
        model, ck = load_model(args.checkpoint, device)
        size = args.img_size or ck['img_size']
        in_ch = 3 + EXTRA_CHANNELS[ck.get('extra_input', 'none')]   # препроцессинг доп. каналов (cv2) в FLOPs не входит, он ≈ 0.05 GFLOPs
    else:
        from train import build_model, parse_decoder
        model = build_model(args.encoder, weights=None, aux_head=True, decoder_channels=parse_decoder(args.decoder_channels)).to(device)
        size = args.img_size or 256
    res = measure(model, size, args.iters, device, in_ch=in_ch)
    print(f'{res["device"]}: img_size={size}  GFLOPs={res["gflops"]:.2f} (limit 100)  latency={res["latency_ms"]:.2f} ms (limit 50, H100 быстрее A4000 ~×3)')
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
