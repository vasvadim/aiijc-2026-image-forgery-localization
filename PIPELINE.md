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

**Аугментации** — `--aug v1` (по умолчанию `none`, только hflip): на исходном разрешении rescale
вниз-вверх (p 0.3), JPEG q60–100 (p 0.5), blur (p 0.15), noise (p 0.15); `--aug v2` — мягче, под
распределение теста: rescale p 0.2, JPEG q75–100 p 0.3, blur/noise p 0.1 (~55 % картинок нетронуты).
Пресеты в `data.AUG_PRESETS`.
Мотивация и замеры — RESEARCH.md §3.5 (пункт 1 плана). Аугментации применяются и к негативам.

**DINO-энкодер** — `--encoder dino:<timm-имя>` (например `dino:vit_small_patch16_dinov3.lvd1689m`), модель в
`src/dino_seg.py`: ViT + лёгкий FPN-декодер до 1/4 + image-level голова. Веса DINOv3 качаются с HF
через timm (на кластере работает). FLOPs с декодером: 704² — 91.8, 736² — 100.3 (не влезает).
`--decoder-channels` игнорируется. Рекомендуемый lr для ViT ниже, чем у EfficientNet: 3e-5…5e-5.

**Mask-mixing** — `--mask-mix P`: для строк с оригиналом того же размера с вероятностью P собирается новый
позитив из выровненной пары (src, chng, GT): режим crop (70 %) — в оригинал вставляется кусок правки внутри
случайного региона R (прямоугольник/эллипс с центром в GT), маска = R ∩ GT, размер R подбирается под
целевую площадь из эмпирического распределения ids9; режим hole (30 %) — в правку возвращается оригинал
внутри R, маска = GT \ R. Даёт маски произвольной формы и мелкие фрагменты (RESEARCH §3.4-E). Работает
только на доменах с src (coco/raise/openimages). Примеры: reports/maskmix_examples.jpg.

**Mask-mixing v2 (07.09)** — три доп. флага к `--mask-mix P`: `--mix-areas ids9|small|mix` (целевые площади
crop-режима: эмпирика ids9 / лог-равномерно 0.3–5 % кадра / 60 % small + 40 % ids9), `--mix-feather F`
(растушёвка края вставки, σ ~ U(0, F) px, чтобы сеть не училась на жёсткую границу склейки),
`--mix-shapes box,ellipse,poly` (формы региона). Мотивация: RESEARCH §9 — дефицит Dice в масках 1–5 % кадра,
v1 с площадями ids9 (медиана 10 %) мелкие маски не сдвинул. Примеры: `reports/maskmix_v2_examples.jpg`.
Рекомендуемый запуск с контроля ep20:
`--mask-mix 0.5 --mix-areas mix --mix-feather 4 --mix-shapes box,ellipse,poly --init-weights <best_proxy ep20> --lr 3e-5 --scheduler cosine --epochs 12 --patience 5`.
Неизменённые маски остаются: микс применяется к 50 % строк с оригиналом, ids9 и негативы не трогаются.

**Edge-loss** — `--edge-weight W`: к BCE+Dice добавляется W·BCE на полосе ~5 px вокруг контура GT
(dilate−erode), мотивация — разбор ошибок (половина дефицита Dice на границах). 0 — выключено.

**Тёплый старт** — `--init-weights <best_proxy.pth>`: только веса модели, оптимизатор/scheduler новые
(в отличие от `--resume`, который восстанавливает всё состояние). Для дообучения с другим лоссом/lr.
`--new-lr-mult M` — параметры, которых нет в чекпоинте (новые модули), идут отдельной группой
оптимизатора с lr × M (cosine масштабирует обе группы).

**Patch-голова** — `--patch-head` (src/forensic.py, RESEARCH §8.3): Conv1×1 на самом глубоком признаке
энкодера (32×32 при 1024²) отвечает «в патче есть правка»; loss + `--patch-weight`·BCE с мягкой целью
clip(доля GT в патче / 0.1, 0, 1). На валидации и инференсе вероятность сливается: p = max(p_pix, γ·up(p_patch)),
γ = `--patch-gamma` (по умолчанию 0.8; пишется в чекпоинт, `predict.py --patch-gamma` переопределяет, 0 —
без слияния). В log.jsonl `proxy` — по слитой вероятности, `proxy.nopatch` — по чисто пиксельной. ≈ 0 FLOPs.

**Доп. входные каналы** — `--extra-input hp3`: к RGB добавляются SRM 5×5, медианный остаток (k3) и локальный
std остатка (окно 16), считаются в Dataset после ресайза (одинаково train/инференс; ≈ 0.05 GFLOPs cv2 вне
модели). В энкодере отдельная свёртка `conv_stem.extra` (zero-init) прибавляется к выходу conv_stem, веса
контроля грузятся как есть (`forensic.adapt_state_dict`). FLOPs модели 88.5 → 89.2. Чекпоинт хранит
`extra_input`, `predict.py`/`measure.py` подхватывают автоматически.

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
Флаги: `--loss bce|bce_dice`, `--scheduler none|cosine`, `--limit N` (отладка), `--full-every 0` (без full),
`--patience 5` — стоп, если критерий `--select-by` не улучшался 5 эпох подряд (D-12; 0 = выключить).
`--aux-head` — image-level голова «есть ли правка» (smp `aux_params`, avg-pool + dropout); loss += `--cls-weight`·BCE.
С головой proxy-AIC считается **с гейтингом** при τ=0.5 (p_img < τ → пустая маска), рядом логируется ungated
и качество классификатора (`cls/acc`, `cls/tpr`, `cls/tnr`). `--neg-src N` — добавить N src-оригиналов
train-сцен как чистые примеры (маска пустая, label 0); holdout-сцены не задействованы (group-split).
`predict.py --tau τ` применяет гейтинг и пишет `img_probs.csv`; `tune_threshold.py` при наличии `img_probs.csv`
перебирает t × a_min × τ и пишет `tau` в `threshold.json`. `--rebuild-scheduler` — при resume построить
cosine заново на `--epochs` (нужно, если горизонт изменился).

**Продолжение прерванного обучения.** `last.pth` хранит полное состояние (веса, Adam с моментами,
scheduler, GradScaler, эпоха, best, RNG, id W&B-рана). Та же команда + `--resume auto` продолжает
со следующей эпохи в тот же W&B-ран. `best_*.pth` — только веса.

**W&B.** Включён по умолчанию (`--no-wandb` выключает), проект `aic2026-detective`, имя рана = `--run-name`.
Один раз на кластере: `pip install -r requirements.txt` и `wandb login` (ключ у Вадима, в репо не хранить).
Логируются: train/val loss и их разрыв, proxy Dice/FPR по пулам/AIC, full-метрика, Dice по доменам и по
площади GT, доля пустых прогнозов на позитивах, медиана площади на негативах, lr, норма градиента
(по шагам и по эпохе), память GPU, время эпохи, `epochs_since_best`, панель из 8 фиксированных картинок
с GT и предсказанием. `tune_threshold.py` и `predict.py --role val-report/control` дописывают свои числа
в summary того же рана.

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
