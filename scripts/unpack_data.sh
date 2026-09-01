#!/bin/bash
# Распаковывает архивы из data/raw в data/ (stage1/, test_stage1/, nanobanana/) и проверяет размеры.
# Запускать на кластере: bash scripts/unpack_data.sh   (лог: ~/aic2026/logs/unpack.log)
set -euo pipefail
DATA=${AIC_DATA_DIR:-/home/vasilenkova/aic2026/data}
RAW="$DATA/raw"
LOG="$DATA/../logs/unpack.log"
cd "$DATA"

# ожидаемые размеры (байты) с HF — контроль полноты скачивания
declare -A EXPECT=( [train_stage1.zip]=39045572403 [test_stage1.zip]=591541868 [nanobanana.zip]=16857702684 )
for f in "${!EXPECT[@]}"; do
  sz=$(stat -c %s "$RAW/$f" 2>/dev/null || echo 0)
  if [ "$sz" != "${EXPECT[$f]}" ]; then echo "!! $f: размер $sz, ожидалось ${EXPECT[$f]} — скачивание не завершено" | tee -a "$LOG"; exit 1; fi
done
echo ">> размеры архивов совпадают" | tee -a "$LOG"

for f in test_stage1.zip train_stage1.zip nanobanana.zip; do
  echo ">> $(date) unzip $f" | tee -a "$LOG"
  unzip -q -n "$RAW/$f" -d "$DATA" 2>>"$LOG"
done
echo ">> $(date) готово" | tee -a "$LOG"
ls -la "$DATA"
du -sh "$DATA"/stage1 "$DATA"/test_stage1 "$DATA"/nanobanana 2>/dev/null
find "$DATA"/stage1 "$DATA"/test_stage1 "$DATA"/nanobanana -maxdepth 2 -type d | head -30
