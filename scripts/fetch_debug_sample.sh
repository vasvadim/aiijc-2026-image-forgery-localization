#!/bin/bash
# Выгружает на локальную машину маленькую отладочную выборку: test_stage1 целиком + N случайных
# train-примеров (img+mask+src если есть) с урезанным train.csv. Локально: ./data (в .gitignore).
# Запускать локально (Git Bash), нужен алиас `aic` в ~/.ssh/config:  bash scripts/fetch_debug_sample.sh 200
set -euo pipefail
N=${1:-200}
REMOTE=/home/vasilenkova/aic2026/data
LOCAL="$(dirname "$0")/../data"
mkdir -p "$LOCAL"

echo ">> test_stage1"
rsync -a --info=progress2 aic:"$REMOTE/test_stage1/" "$LOCAL/test_stage1/"

echo ">> $N train-примеров"
ssh aic "cd $REMOTE && python3 - <<'EOF'
import csv, random, os
random.seed(42)
rows = list(csv.DictReader(open('stage1/train.csv', encoding='utf-8')))
sample = random.sample(rows, min($N, len(rows)))
os.makedirs('/tmp/aic_sample', exist_ok=True)
with open('/tmp/aic_sample/train.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(sample)
with open('/tmp/aic_sample/files.txt', 'w') as f:
    for r in sample:
        for k in ('orgl_img_path', 'chng_img_path', 'gt_path'):
            if r.get(k): f.write(r[k].strip().replace('\\\\', '/') + '\n')
print('ok', len(sample))
EOF"
mkdir -p "$LOCAL/stage1"
rsync -a --info=progress2 --files-from=<(ssh aic cat /tmp/aic_sample/files.txt) aic:"$REMOTE/" "$LOCAL/"
scp -q aic:/tmp/aic_sample/train.csv "$LOCAL/stage1/train.csv"
echo ">> готово: $LOCAL"
du -sh "$LOCAL"/*
