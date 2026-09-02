#!/bin/bash
# Синхронизация offline-ранов W&B с кластера в облако через локальную машину.
# Кластер не имеет доступа к api.wandb.ai (403, гео-блок), поэтому train.py пишет ран в offline-режиме
# (WANDB_MODE=offline) в <run_dir>/wandb/offline-run-*, а этот скрипт (локально, Git Bash, `wandb login` сделан)
# забирает каталоги и делает `wandb sync`. Повторный запуск досылает только новое.
#   bash scripts/wandb_sync.sh [<login>] [<run_name>]      # без аргументов — все раны всех логинов
set -uo pipefail
LOGIN=${1:-}
RUN=${2:-}
REMOTE=/home/vasilenkova/aic2026/runs
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/runs/_wandb_sync"
mkdir -p "$LOCAL"
PATTERN="$REMOTE/${LOGIN:-*}/${RUN:-*}/wandb/offline-run-*"
for i in 1 2 3 4 5; do
  timeout 300 ssh -o ConnectTimeout=12 aic "cd / && tar cf - --exclude='*/logs' --exclude='*/tmp' \$(ls -d $PATTERN 2>/dev/null | sed 's#^/##') 2>/dev/null" > "$LOCAL/sync.tar" && break
  echo "ssh retry $i"; sleep 8
done
if [ ! -s "$LOCAL/sync.tar" ]; then echo "нет offline-ранов или ssh недоступен"; exit 1; fi
tar xf "$LOCAL/sync.tar" -C "$LOCAL" 2>/dev/null; rm -f "$LOCAL/sync.tar"
for d in $(find "$LOCAL" -type d -name 'offline-run-*'); do
  echo ">> sync $d"
  timeout 300 wandb sync --include-offline "$d" 2>&1 | grep -i "syncing\|error" | tail -2
done
