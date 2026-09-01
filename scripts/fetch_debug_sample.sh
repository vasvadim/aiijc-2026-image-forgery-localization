#!/bin/bash
# Выгружает на локальную машину маленькую отладочную выборку: test_stage1 целиком + N случайных
# train-примеров (img+mask+src если есть) с урезанным train.csv. Локально: ./data (в .gitignore).
# Запускать локально (Git Bash), нужен алиас `aic` в ~/.ssh/config:  bash scripts/fetch_debug_sample.sh 200
# Без rsync (в Git Bash его нет) — упаковываем на кластере в tar и тянем одним потоком.
set -euo pipefail
N=${1:-200}
REMOTE=/home/vasilenkova/aic2026/data
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$LOCAL"

echo ">> формирую выборку на кластере ($N train-примеров + test_stage1)"
ssh aic "cd $REMOTE && python3 - <<'EOF'
import csv, random, os, tarfile
random.seed(42)
rows = list(csv.DictReader(open('stage1/train.csv', encoding='utf-8')))
sample = random.sample(rows, min($N, len(rows)))
os.makedirs('/tmp/aic_sample/stage1', exist_ok=True)
with open('/tmp/aic_sample/stage1/train.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(sample)
with tarfile.open('/tmp/aic_sample/sample.tar', 'w') as t:
    t.add('/tmp/aic_sample/stage1/train.csv', arcname='stage1/train.csv')
    t.add('test_stage1', arcname='test_stage1')
    for r in sample:
        for k in ('orgl_img_path', 'chng_img_path', 'gt_path'):
            p = (r.get(k) or '').strip().replace('\\\\', '/')
            if p and os.path.exists(p): t.add(p, arcname=p)
print('ok', len(sample), os.path.getsize('/tmp/aic_sample/sample.tar') // 2**20, 'MiB')
EOF"
echo ">> скачиваю"
ssh aic "cat /tmp/aic_sample/sample.tar" | tar xf - -C "$LOCAL"
ssh aic "rm -rf /tmp/aic_sample"
echo ">> готово: $LOCAL"
du -sh "$LOCAL"/*
