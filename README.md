<h1 align="center">🔍 AI Challenge 2026 (AIIJC) — Image Manipulation Localization</h1>

<p align="center">
  Task A «Цифровой детектив: поймай генеративку» (MWS AI × VisionLabs): find the region of a photo that was edited by a generative model.<br/>
  Team <b>«Амнямы» / OnlyOmNoms AI</b> (Vadim Vasilenko, <a href="https://huggingface.co/Kseniiashk">Kseniia Shkuleva</a>), student track, September–October 2026 · <b>finalists, Computer Vision track</b>.
</p>

<p align="center">
  <a href="https://huggingface.co/OnlyOmNoms/aiijc_2026_detective"><img alt="Weights" src="https://img.shields.io/badge/%F0%9F%A4%97%20weights%20%26%20assets-OnlyOmNoms%2Faiijc__2026__detective-FFD21E"></a>
  <a href="https://huggingface.co/spaces/Kseniiashk/detective"><img alt="Demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20demo-Kseniiashk%2Fdetective-FFD21E"></a>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Metric" src="https://img.shields.io/badge/metric-AIC%20score-informational">
  <img alt="Budget" src="https://img.shields.io/badge/budget-%E2%89%A4%20100%20GFLOPs-orange">
</p>

> **This repository = the main stage (September 2026):** training and validation code, the research notes and the decision log.
> **The final stage (October 2026)** lives on Hugging Face: [weights, splits, thresholds and logs](https://huggingface.co/OnlyOmNoms/aiijc_2026_detective) and the [interactive demo service](https://huggingface.co/spaces/Kseniiashk/detective).

## Task

- **Input:** an RGB image, possibly with a subtle local AI edit (an added scratch on a car, a spoiled dish, a forged receipt, an inserted or removed object) or with no edit at all.
- **Output:** a binary mask of the same size (0 = original, 255 = edited).
- **Metric — AIC score:** the harmonic mean of the per-image Dice on positive images and (1 − FPR) on clean images, where an image counts as a false alarm if the predicted mask covers ≥ 1 % of the frame. A model that fires on every clean image scores 0 regardless of Dice, so the binarisation threshold and the minimum-area rule are tuned directly for AIC.
- **Constraint:** ≤ 100 GFLOPs per image.

## Final solution (October 2026)

| Component | What |
|---|---|
| Model | **U-Net with an EfficientNet-B5 encoder**, a uniform **weight soup of 14 fine-tunes** (`final/soup_f14.pth`), 98.9 GFLOPs on the worst test frame |
| Training | students warm-started from the main-stage B5, **knowledge distillation from a ViT-L teacher**, several student stages (`final/stages/`), mask-mixing and compression augmentations |
| Inference | area cap of 0.79 MP with isotropic downscale, padding to a multiple of 32, fp16 forward, probabilities resized to the original frame (`model_code/aic_infer.py` in the demo) |
| Post-processing | threshold **0.3**, image-level gate at **0.99** (the mask is cleared when the image head says "clean"), hedge: top 0.9 % of pixels instead of an empty mask |
| Demo service | FastAPI + web UI: verdict and probability map, mask overlay and on-the-fly threshold, **edit-type classifier** (DINOv3 ViT-B/16: 9 edit types, the tool used, what was there before), CLIP object naming, **LaMa "how it may have looked"** reconstruction, forensics (ELA, noise, JPEG grid, EXIF / generator traces, SHA-256), robustness stress test, comparison with ground truth, batch mode with `submission.zip`, REST API |

## Main stage (this repository, September 2026)

| Component | What we did |
|---|---|
| Model | U-Net with an **EfficientNet-B4 encoder at 1024²**, light decoder, auxiliary image-level head (`src/train.py`, `src/data.py`); EfficientNet-B5 by the end of the stage |
| Loss / schedule | BCE + Dice, AMP, cosine schedule, warm-started fine-tunes with separate learning rates for new modules |
| Augmentation | rescale / JPEG re-compression / blur / noise presets matched to the test distribution; **mask-mixing**: new positives synthesised from aligned (original, edited, GT) triplets by cropping an edit into the original or restoring a hole, giving masks of arbitrary shape and size |
| Forensic signals | high-pass inputs (SRM filters, median residual, local std) as extra channels (`src/forensic.py`); JPEG-quality probes per domain |
| Small masks | patch-classification head for free-form edits covering 1–5 % of the frame, the main source of lost Dice |
| Explored and dropped | DINOv3 ViT-S/16 + FPN decoder (`src/dino_seg.py`): strong but 161 GFLOPs at 704², over the budget; edge loss (neutral); Tversky / recall-oriented losses |
| Validation | holdout scenes with three negative pools (empty-GT train images, untouched originals, Gemini-generated controls); the local score tracked the leaderboard within ±0.006 after two calibration submissions (`VALIDATION.md`) |
| Threshold tuning | `src/tune_threshold.py` optimises the binarisation threshold and minimum area for AIC, not Dice |

Leaderboard progress during the main stage: organisers' baseline reproduced at **0.589** → EfficientNet-B4 @ 1024² **0.915** → best documented validation AIC **0.947**.

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

## Run (main-stage code)

```bash
bash scripts/setup_env.sh                       # conda env, requirements
export AIC_DATA_DIR=/path/to/data
CUDA_VISIBLE_DEVICES=0 python src/train.py --run-name b4_1024 --encoder efficientnet-b4 --img-size 1024 \
    --epochs 24 --batch-size 8 --lr 1e-4 --loss bcedice --aug v2 --mask-mix 0.3 --amp
python src/tune_threshold.py --run b4_1024       # threshold + min-area for AIC
python src/predict.py --run b4_1024 --out submission/
```

See `PIPELINE.md` for the full set of flags. Final-stage weights and the inference code that reproduces the leaderboard masks are on Hugging Face (links above).

---

<p align="center">Team «Амнямы» · <a href="https://github.com/vasvadim">Vadim Vasilenko</a> · <a href="https://huggingface.co/Kseniiashk">Kseniia Shkuleva</a> · AIIJC 2026</p>
