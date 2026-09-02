#!/bin/bash
# Зеркалирование ранов обучения в W&B через локальную машину (Git Bash, `wandb login` сделан).
# Кластер не имеет доступа к api.wandb.ai (403, гео-блок), а `wandb sync` offline-ранов отсюда зависает,
# поэтому забираем с кластера обычные файлы run_dir (log.jsonl, steps.jsonl, panel/, best.json, threshold.json,
# metrics.json ролей, flops.json) и пишем их в онлайн-ран скриптом scripts/wandb_mirror.py под тем же id
# (wandb_id.txt). Идемпотентно: досылается только новое.
#   bash scripts/wandb_sync.sh <login> <run_name>
#   bash scripts/wandb_sync.sh <login>              # все раны логина
set -uo pipefail
LOGIN=${1:?login на кластере}
RUN=${2:-}
REMOTE=/home/vasilenkova/aic2026/runs/$LOGIN
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/runs/_wandb_sync/$LOGIN"
mkdir -p "$LOCAL"
FILES="log.jsonl steps.jsonl config.json wandb_id.txt best.json threshold.json flops.json panel val-tune/metrics.json val-report/metrics.json control/metrics.json"
if [ -z "$RUN" ]; then
  RUNS=$(timeout 40 ssh -o ConnectTimeout=12 aic "ls -d $REMOTE/*/log.jsonl 2>/dev/null | xargs -n1 dirname | xargs -n1 basename")
else
  RUNS=$RUN
fi
for r in $RUNS; do
  mkdir -p "$LOCAL/$r"
  ok=0
  for i in 1 2 3 4 5; do
    timeout 300 ssh -o ConnectTimeout=12 aic "cd $REMOTE/$r 2>/dev/null && tar cf - \$(ls -d $FILES 2>/dev/null)" > "$LOCAL/$r/sync.tar" && ok=1 && break
    echo "ssh retry $i"; sleep 8
  done
  if [ $ok = 0 ] || [ ! -s "$LOCAL/$r/sync.tar" ]; then echo "$r: ssh недоступен или файлов нет"; rm -f "$LOCAL/$r/sync.tar"; continue; fi
  tar xf "$LOCAL/$r/sync.tar" -C "$LOCAL/$r" 2>/dev/null; rm -f "$LOCAL/$r/sync.tar"
  echo ">> mirror $r"
  timeout "${SYNC_TIMEOUT:-600}" python "$(dirname "$0")/wandb_mirror.py" --dir "$LOCAL/$r" --run-name "$r" 2>&1 | grep -v "^wandb: " | tail -3
done
