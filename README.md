<h1 align="center">🔍 AI Challenge 2026 (AIIJC) — Image Manipulation Localization</h1>

<p align="center">
  Task A «Цифровой детектив: поймай генеративку» (MWS AI × VisionLabs): find the region of a photo that was edited by a generative model.<br/>
  Team of two, student track, September 2026 · <b>finalist, Computer Vision track</b>.
</p>

<p align="center">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Segmentation" src="https://img.shields.io/badge/task-forgery%20segmentation-blueviolet">
  <img alt="Metric" src="https://img.shields.io/badge/metric-AIC%20score-informational">
  <img alt="Budget" src="https://img.shields.io/badge/budget-%E2%89%A4%20100%20GFLOPs-orange">
</p>

## Task

- **Input:** an RGB image, possibly with a subtle local AI edit (an added scratch on a car, a spoiled dish, a forged receipt, an inserted or removed object) or with no edit at all.
- **Output:** a binary mask of the same size (0 = original, 255 = edited).
- **Metric — AIC score:** the harmonic mean of the per-image Dice on positive images and (1 − FPR) on clean images, where an image counts as a false alarm if the predicted mask covers ≥ 1 % of the frame. A model that fires on every clean image scores 0 regardless of Dice, so the binarisation threshold and the minimum-area rule are tuned directly for AIC.
- **Constraint:** ≤ 100 GFLOPs per image.

## Approach

| Component | What we did |
|---|---|
| Model | U-Net with an **EfficientNet-B4 encoder at 1024²**, light decoder, auxiliary image-level head (`src/train.py`, `src/data.py`) |
| Loss / schedule | BCE + Dice, AMP, cosine schedule, warm-started fine-tunes with separate learning rates for new modules |
| Augmentation | rescale / JPEG re-compression / blur / noise presets matched to the test distribution; **mask-mixing**: new positives synthesised from aligned (original, edited, GT) triplets by cropping an edit into the original or restoring a hole, giving masks of arbitrary shape and size |
| Forensic signals | high-pass inputs (SRM filters, median residual, local std) as extra channels (`src/forensic.py`); JPEG-quality probes per domain |
| Small masks | patch-classification head for free-form edits covering 1–5 % of the frame, the main source of lost Dice |
| Explored and dropped | DINOv3 ViT-S/16 + FPN decoder (`src/dino_seg.py`): strong but 161 GFLOPs at 704², over the budget; edge loss (neutral); Tversky / recall-oriented losses |
| Validation | holdout scenes with three negative pools (empty-GT train images, untouched originals, Gemini-generated controls); the local score tracked the leaderboard within ±0.006 after two calibration submissions (`VALIDATION.md`) |
| Threshold tuning | `src/tune_threshold.py` optimises the binarisation threshold and minimum area for AIC, not Dice |

Progress on the leaderboard: organisers' baseline reproduced at **0.589** → EfficientNet-B4 @ 1024² **0.915**; best documented validation AIC **0.947**.

## Repository

| Path | Contents |
|---|---|
| `HACKATHON.md` | task statement, data, metric, constraints, submission format |
| `PIPELINE.md` | how to train, tune the threshold, predict and build a submission |
| `VALIDATION.md`, `DECISIONS.md`, `RESEARCH.md`, `experiments.md` | validation protocol, decision log, literature and data research, table of all runs |
| `TEAM.md` | how the two of us shared the cluster, data and branches |
| `src/` | `train.py`, `predict.py`, `data.py`, `dino_seg.py`, `forensic.py`, `metric.py`, `tune_threshold.py`, `make_split.py`, `measure.py` (GFLOPs) |
| `scripts/` | EDA, error analysis, calibration probes, cluster helpers, W&B sync |
| `splits/`, `reports/` | validation splits and JSON reports of the analyses |
| `reference/baseline/` | the organisers' baseline |

## Run

```bash
bash scripts/setup_env.sh                       # conda env, requirements
export AIC_DATA_DIR=/path/to/data
CUDA_VISIBLE_DEVICES=0 python src/train.py --run-name b4_1024 --encoder efficientnet-b4 --img-size 1024 \
    --epochs 24 --batch-size 8 --lr 1e-4 --loss bcedice --aug v2 --mask-mix 0.3 --amp
python src/tune_threshold.py --run b4_1024       # threshold + min-area for AIC
python src/predict.py --run b4_1024 --out submission/
```

See `PIPELINE.md` for the full set of flags.

---

<p align="center"><a href="https://github.com/vasvadim">Vadim Vasilenko</a> and teammate · AIIJC 2026</p>
