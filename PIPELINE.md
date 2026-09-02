# PIPELINE.md — как запускать обучение, подбор порога, инференс и сабмит

> Для людей и агентов. Все скрипты в `src/`, все читают `AIC_DATA_DIR`, пишут в
> `AIC_RUNS_DIR/<login>/<run_name>/` (по умолчанию `<repo>/../runs`, на кластере `~/aic2026/runs`).
> Схема валидации — [VALIDATION.md](VALIDATION.md), решения — [DECISIONS.md](DECISIONS.md).

## 0. Подготовка (один раз в сессии)

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate /home/vasilenkova/aic2026/envs/train
export AIC_DATA_DIR=/home/vasilenkova/aic2026/data
cd ~/aic2026/code && git pull
bash scripts/gpu_free.sh          # выбрать свободную карту
tmux new -s <имя>                 # обучение только внутри tmux/nohup
```

## 1. Обучение — `src/train.py`

```bash
CUDA_VISIBLE_DEVICES=6 python src/train.py --run-name 20260902_baseline_r34_256 \
    --encoder resnet34 --img-size 256 --epochs 15 --batch-size 32 --lr 1e-4 --loss bce \
    --select-by proxy --full-every 1 --workers 8 --amp
```

**Несколько GPU (DDP)** — тот же скрипт через `torchrun`; `--batch-size` задаётся на каждую карту,
lr автоматически не масштабируется (эффективный батч = batch × число карт):

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 src/train.py --run-name <имя> --batch-size 32 --workers 8 --amp ...
```

Валидацию и full-метрику считает rank 0, остальные ждут. Замер на A4000 (UNet-R34, 256², AMP):
шаг модели 289 img/s на карту, загрузчик 727 img/s на 10 воркерах → одна карта ≈ 6 мин/эпоху,
две ≈ 3.3 мин. Свободные карты смотреть через `scripts/gpu_free.sh`; больше двух — по договорённости (TEAM.md §4).

Каждую эпоху в `log.jsonl` пишутся `train_loss`, `val_loss` (весь val-tune), `proxy` (Dice/FPR/AIC
на тензорах, proxy-строки) и `full` (metric.py на PNG, те же строки, t=0.5). Чекпоинты:
`last.pth`, `best_loss.pth`, `best_proxy.pth`, `best_full.pth`, `best.pth` (= по `--select-by`).
Флаги: `--loss bce|bce_dice`, `--scheduler none|cosine`, `--limit N` (отладка), `--full-every 0` (без full).

## 2. Вероятности на val-tune и подбор порога — `predict.py --save-probs` + `tune_threshold.py`

```bash
R=~/aic2026/runs/$USER/20260902_baseline_r34_256
CUDA_VISIBLE_DEVICES=6 python src/predict.py --checkpoint $R/best.pth --role val-tune --out $R/val-tune --save-probs --amp
python src/tune_threshold.py --probs-dir $R/val-tune/probs --out $R/threshold.json
```

`predict.py` по роли split сам вызывает `metric.py` и пишет `metrics.json` (при t=0.5, a_min=0).
`tune_threshold.py` печатает сетку AIC по t × a_min и выбирает центр плато (D-05) → `threshold.json`.

## 3. Отчёт на val-report и контроль — только с замороженными t, a_min

```bash
T=$(python -c "import json;print(json.load(open('$R/threshold.json'))['t'])")
A=$(python -c "import json;print(json.load(open('$R/threshold.json'))['a_min'])")
CUDA_VISIBLE_DEVICES=6 python src/predict.py --checkpoint $R/best.pth --role val-report --out $R/val-report --t $T --a-min $A --amp
CUDA_VISIBLE_DEVICES=6 python src/predict.py --checkpoint $R/best.pth --role control    --out $R/control    --t $T --a-min $A --amp
```

val-report смотрим один раз перед сабмитом и записываем в `experiments.md`. Ничего по нему не подбираем.

## 4. Тест и сабмит

```bash
CUDA_VISIBLE_DEVICES=6 python src/predict.py --checkpoint $R/best.pth --test --out $R/test --t $T --a-min $A --amp --zip
python src/measure.py --checkpoint $R/best.pth --out $R/flops.json
```

`$R/test/submission.zip` — то, что грузится на платформу (внутри `submission.csv` + `predictions/`).
Перед загрузкой: строка в `experiments.md` с val-tune, val-report, t, a_min, FLOPs; сообщение в чат команды.

## 5. Ручная проверка метрики

```bash
python src/metric.py --selftest
python src/metric.py --submission $R/val-report/submission.csv --split splits/stage1_v1.csv.gz --role val-report
```
