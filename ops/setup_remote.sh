#!/bin/bash
# =============================================================================
# ops/setup_remote.sh
# =============================================================================
# One-time setup script for your remote GB10 (DGX Spark) machine.
# Run this once after SSH-ing in.
#
# Usage:
#   ssh user@your-gb10
#   chmod +x ops/setup_remote.sh
#   ./ops/setup_remote.sh
# =============================================================================

set -e   # Exit on any error
set -o pipefail

echo "============================================================"
echo "  Text-to-SQL Pipeline – Remote Setup (GB10 / DGX Spark)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Check NVIDIA driver and CUDA
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Checking GPU setup..."
nvidia-smi || { echo "ERROR: nvidia-smi failed. Is the NVIDIA driver installed?"; exit 1; }
echo "✓ NVIDIA driver OK"

# Check CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "  CUDA Version: $CUDA_VERSION"

if [[ $(echo "$CUDA_VERSION >= 12.0" | bc -l) -eq 0 ]]; then
    echo "WARNING: CUDA < 12.0 detected. Training Hub requires CUDA >= 12.0"
    echo "  Please update your NVIDIA driver."
fi

# ---------------------------------------------------------------------------
# 2. Check Docker and NVIDIA Container Toolkit
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Checking Docker setup..."
docker --version || { echo "ERROR: Docker not installed"; exit 1; }
echo "✓ Docker OK"

# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi 2>/dev/null || {
    echo ""
    echo "NVIDIA Container Toolkit not found. Installing..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "✓ NVIDIA Container Toolkit installed"
}
echo "✓ NVIDIA Container Toolkit OK"

# ---------------------------------------------------------------------------
# 3. Create data directories
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Creating data directories..."
sudo mkdir -p /data/hf_cache
sudo chmod 777 /data/hf_cache

mkdir -p \
    ./data/raw \
    ./data/generated \
    ./data/final \
    ./data/final/checkpoints \
    ./data/evaluation \
    ./logs/tensorboard

echo "✓ Directories created"
echo "  HuggingFace cache: /data/hf_cache (change in docker-compose.yml if needed)"

# ---------------------------------------------------------------------------
# 4. Create local config from template
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Setting up local config..."
if [ ! -f config/pipeline_config.local.yaml ]; then
    cp config/pipeline_config.yaml config/pipeline_config.local.yaml
    echo "✓ Created: config/pipeline_config.local.yaml"
    echo ""
    echo "  ACTION REQUIRED: Edit config/pipeline_config.local.yaml"
    echo "  Fill in your API key and set teacher.backend."
    echo ""
    echo "  Example (Anthropic):"
    echo "    teacher:"
    echo "      backend: anthropic"
    echo "      anthropic:"
    echo "        api_key: YOUR_ANTHROPIC_API_KEY"
else
    echo "  config/pipeline_config.local.yaml already exists – skipping"
fi

# ---------------------------------------------------------------------------
# 5. Build Docker images
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Building Docker images..."
echo "  This may take several minutes on first run..."

docker compose -f docker/docker-compose.yml build sdg
echo "  ✓ SDG image built"

docker compose -f docker/docker-compose.yml build training
echo "  ✓ Training image built"

# ---------------------------------------------------------------------------
# 6. Verify setup with a dry run
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Verifying setup with dry run..."
docker compose -f docker/docker-compose.yml run --rm sdg \
    python sdg_pipeline/run_sdg.py \
    --config config/pipeline_config.local.yaml \
    --dry-run \
    --n-samples 5 2>&1 | tail -20 || echo "  (dry run requires seed data – run prepare_data.py first)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup complete! Next steps:"
echo "============================================================"
echo ""
echo "  1. Edit your config (fill in API key):"
echo "     nano config/pipeline_config.local.yaml"
echo ""
echo "  2. Download and prepare seed data:"
echo "     docker compose -f docker/docker-compose.yml run --rm sdg \\"
echo "       python data_pipeline/prepare_data.py --config config/pipeline_config.local.yaml"
echo ""
echo "  3. Run SDG pipeline:"
echo "     docker compose -f docker/docker-compose.yml run --rm sdg \\"
echo "       python sdg_pipeline/run_sdg.py --config config/pipeline_config.local.yaml"
echo ""
echo "  4. Mix datasets:"
echo "     docker compose -f docker/docker-compose.yml run --rm sdg \\"
echo "       python data_pipeline/mix_datasets.py"
echo ""
echo "  5. Run training:"
echo "     docker compose -f docker/docker-compose.yml run --rm training \\"
echo "       python training_pipeline/train.py --config config/pipeline_config.local.yaml"
echo ""
echo "  6. Evaluate:"
echo "     docker compose -f docker/docker-compose.yml run --rm training \\"
echo "       python evaluation/evaluate.py \\"
echo "         --model-path /app/data/final/checkpoints/lora \\"
echo "         --use-adapter"
echo ""
echo "  Optional: Use local vLLM as teacher model (instead of API):"
echo "     docker compose -f docker/docker-compose.yml --profile vllm up -d vllm"
echo "     # Set teacher.backend: vllm_local in your config"
echo ""
