#!/bin/bash
# Печатает GPU по возрастанию занятости памяти: index, used MiB, total MiB, util %.
# Использование: bash scripts/gpu_free.sh        → таблица
#                bash scripts/gpu_free.sh -n 1   → только индекс самой свободной карты
N=""
if [ "$1" == "-n" ]; then N="$2"; fi
LIST=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | sort -t, -k2 -n)
if [ -n "$N" ]; then
  echo "$LIST" | head -n "$N" | cut -d, -f1 | tr -d ' ' | paste -sd,
else
  printf "%-5s %-10s %-10s %-6s\n" idx used_MiB total_MiB util%
  echo "$LIST" | awk -F', ' '{printf "%-5s %-10s %-10s %-6s\n",$1,$2,$3,$4}'
fi
