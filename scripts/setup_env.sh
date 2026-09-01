#!/bin/bash
# Создаёт (или обновляет) ОБЩЕЕ conda-окружение для обучения и регистрирует Jupyter-kernel.
# Запускать на кластере из корня репозитория: bash scripts/setup_env.sh
set -euo pipefail
ENV_DIR=${AIC_ENV_DIR:-/home/vasilenkova/aic2026/envs/train}
source /opt/miniconda3/etc/profile.d/conda.sh

if [ ! -d "$ENV_DIR" ]; then
  echo ">> создаю окружение $ENV_DIR"
  conda create -y -p "$ENV_DIR" python=3.11
fi
conda activate "$ENV_DIR"
python -m pip install -U pip
python -m pip install -r "$(dirname "$0")/../requirements.txt"
python -m ipykernel install --user --name aic2026 --display-name "Python (aic2026)"
python - <<'EOF'
import torch, segmentation_models_pytorch as smp, timm, albumentations, cv2
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("smp", smp.__version__, "timm", timm.__version__, "albumentations", albumentations.__version__, "cv2", cv2.__version__)
EOF
echo ">> готово. Активация: conda activate $ENV_DIR ; kernel: aic2026"
