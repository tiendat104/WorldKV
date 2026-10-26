#!/usr/bin/env bash
# Prepare a fresh RunPod pod (PyTorch 2.8.0 template) for the evaluation harness.
#
# Everything goes on the network volume (/workspace), so it survives the pod:
#   /workspace/WorldKV                         code (eval-harness branch)
#   /workspace/venv                            Python environment (uses the template's PyTorch)
#   /workspace/models/lingbot-world-base-cam   the 4 model parts LingBot-World-Fast needs (~81 GB)
# Steps that were already done are skipped, so the script can be run again.
#
# Usage on the pod:
#   curl -sL https://raw.githubusercontent.com/tiendat104/WorldKV/eval-harness/eval/setup_pod.sh | bash
# or, if the code is already on the volume:
#   bash /workspace/WorldKV/eval/setup_pod.sh

set -euo pipefail
WS=${WORKSPACE:-/workspace}
REPO_URL=${REPO_URL:-https://github.com/tiendat104/WorldKV.git}
BRANCH=${BRANCH:-eval-harness}
VENV=$WS/venv
PY=$VENV/bin/python
MODELS=$WS/models/lingbot-world-base-cam
step() { echo; echo "=== $*"; }

step "1/6 Code"
if [ -d "$WS/WorldKV/.git" ]; then
  git -C "$WS/WorldKV" pull --ff-only || echo "Could not update the code; check 'git status' in $WS/WorldKV"
else
  git clone -b "$BRANCH" "$REPO_URL" "$WS/WorldKV"
fi
git -C "$WS/WorldKV" log -1 --format='code at %h %s'

step "2/6 Python environment ($VENV)"
[ -x "$PY" ] || python -m venv --system-site-packages "$VENV"
grep -v flash_attn "$WS/WorldKV/requirements.txt" > /tmp/requirements_no_flash_attn.txt
"$PY" -m pip install -q -r /tmp/requirements_no_flash_attn.txt -r "$WS/WorldKV/eval/requirements.txt" \
  "huggingface_hub[cli]" ninja packaging
"$PY" -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"

step "3/6 flash_attn"
if "$PY" -c "import flash_attn" 2>/dev/null; then
  echo "already installed"
else
  MAX_JOBS=8 "$PY" -m pip install flash-attn --no-build-isolation
fi

step "4/6 Model files (only the parts LingBot-World-Fast needs; skips files already there)"
mkdir -p "$MODELS"
# One file name per argument: `hf download --include a b c` only keeps the first pattern.
"$VENV/bin/hf" download robbyant/lingbot-world-base-cam \
  models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth configuration.json --local-dir "$MODELS"
"$VENV/bin/hf" download robbyant/lingbot-world-base-cam --include "google/umt5-xxl/*" --local-dir "$MODELS"
"$VENV/bin/hf" download robbyant/lingbot-world-fast --local-dir "$MODELS/lingbot_world_fast"

step "5/6 Check model files"
"$PY" - "$MODELS" <<'EOF'
import glob, os, sys
m = sys.argv[1]
need = {  # file: minimum size in bytes
    "models_t5_umt5-xxl-enc-bf16.pth": 11.3e9,
    "Wan2.1_VAE.pth": 5.0e8,
    "google/umt5-xxl/tokenizer.json": 1e6,
    "google/umt5-xxl/spiece.model": 1e6,
    "lingbot_world_fast/config.json": 1,
    "lingbot_world_fast/diffusion_pytorch_model.safetensors.index.json": 1,
}
bad = [f for f, size in need.items()
       if not os.path.exists(os.path.join(m, f)) or os.path.getsize(os.path.join(m, f)) < size]
shards = glob.glob(os.path.join(m, "lingbot_world_fast", "*.safetensors"))
total = sum(os.path.getsize(s) for s in shards)
if len(shards) != 16 or total < 74e9:
    bad.append(f"lingbot_world_fast shards: {len(shards)} files, {total / 1e9:.1f} GB (expected 16, ~74 GB)")
if bad:
    sys.exit("MISSING OR INCOMPLETE: " + "; ".join(bad))
print(f"all model files present ({len(shards)} shards, {total / 1e9:.1f} GB)")
EOF

step "6/6 Machine checks"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvidia-smi topo -m | head -n "$(( $(nvidia-smi -L | wc -l) + 1 ))"
"$PY" -c "
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
print('flash_attn', flash_attn.__version__, 'works on', torch.cuda.get_device_name())
flash_attn_func(q, q, q)"
cd "$WS/WorldKV"
"$PY" -m pytest eval/tests -q -W ignore 2>&1 | tail -n 1
NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -gt 1 ]; then
  echo "GPU communication test on $NGPU GPUs (default settings):"
  if timeout 120 "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" -m eval.nccl_check 2>&1 \
       | grep -E "^rank [0-9]+:"; then
    echo "-> GPUs can talk to each other: run the runner WITHOUT --no-p2p."
  else
    echo "-> GPU communication hung or failed: run the runner WITH --no-p2p."
  fi
fi

echo
echo "Setup done. Activate the environment with: source $VENV/bin/activate"
