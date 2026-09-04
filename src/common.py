"""common.py — общие утилиты пайплайна: пути, сиды, чтение split/картинок/масок, нормализация.

Все пути к данным — только через AIC_DATA_DIR (TEAM.md §2). Запуски — в AIC_RUNS_DIR
(по умолчанию <repo>/../runs, на кластере это ~/aic2026/runs), внутри <login>/<run_name>/.
"""
from __future__ import annotations

import getpass
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True   # среди src-оригиналов есть обрезанные JPEG (04.09) — читаем, что есть, вместо OSError

REPO = Path(__file__).resolve().parents[1]
GT_THRESH = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def data_dir() -> Path:
    d = os.environ.get('AIC_DATA_DIR')
    if not d:
        raise SystemExit('переменная AIC_DATA_DIR не задана (см. TEAM.md §2)')
    return Path(d)


def runs_dir() -> Path:
    return Path(os.environ.get('AIC_RUNS_DIR', REPO.parent / 'runs')) / getpass.getuser()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_path(version: str) -> Path:
    p = Path(version)
    if p.exists():
        return p
    return REPO / 'splits' / f'stage1_{version}.csv.gz'


def load_split(version: str, role: str | None = None) -> pd.DataFrame:
    """role: train | val-tune | val-report | proxy (in_proxy==1) | control | None (все строки)."""
    df = pd.read_csv(split_path(version), dtype=str, keep_default_na=False)
    if role == 'proxy':
        df = df[df['in_proxy'] == '1']
    elif role:
        df = df[df['role'] == role]
    if df.empty:
        raise SystemExit(f'в split {version} нет строк роли {role}')
    df = df.reset_index(drop=True)
    df['H'] = df['H'].astype(int)
    df['W'] = df['W'].astype(int)
    df['is_neg'] = df['is_neg'].astype(int)
    df['in_proxy'] = df['in_proxy'].astype(int)
    return df


def load_test_df() -> tuple[pd.DataFrame, Path]:
    """test.csv организаторов; пути в нём относительно папки test_stage1 (HACKATHON.md §3.2)."""
    root = data_dir() / 'test_stage1'
    df = pd.read_csv(root / 'test.csv', dtype=str, keep_default_na=False)
    df['img_path'] = df['img_path'].str.strip().str.replace('\\', '/', regex=False)
    return df, root


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGB'))


def load_gt(path: str | Path) -> np.ndarray:
    """GT как bool HxW, бинаризация >128 (VALIDATION.md §1)."""
    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]
    return m > GT_THRESH


def normalize(img_uint8: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 → CxHxW float32 с ImageNet-нормализацией."""
    x = (img_uint8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def parse_size(v) -> tuple[int, int]:
    """'1024x768' | '640' | (w, h) | [w, h] → (w, h). Стороны должны делиться на 32 (UNet)."""
    if isinstance(v, (tuple, list)):
        w, h = int(v[0]), int(v[1])
    else:
        s = str(v).lower().replace('×', 'x')
        w, h = (int(s.split('x')[0]), int(s.split('x')[1])) if 'x' in s else (int(s), int(s))
    assert w % 32 == 0 and h % 32 == 0, f'размер {w}x{h} должен делиться на 32'
    return w, h


def save_binary_png(mask_bool: np.ndarray, path: str | Path) -> None:
    Image.fromarray((mask_bool.astype(np.uint8) * 255), mode='L').save(path)


def wandb_update_summary(run_dir: str | Path, data: dict) -> None:
    """Дописать числа (порог, val-report, LB) в summary W&B-рана обучения, если он был (run_dir/wandb_id.txt)."""
    try:
        wid = (Path(run_dir) / 'wandb_id.txt').read_text().strip()
        import wandb
        run = wandb.init(id=wid, resume='must', project=os.environ.get('WANDB_PROJECT', 'aic2026-detective'), dir=str(run_dir))
        run.summary.update(data)
        run.finish()
    except Exception as e:  # noqa: BLE001
        print(f'wandb summary не обновлён: {e!r}')
