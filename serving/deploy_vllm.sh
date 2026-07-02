#!/bin/bash
# =============================================================================
# serving/deploy_vllm.sh
# =============================================================================
# Merged das LoRA-Modell und startet einen vLLM Server.
#
# Usage:
#   ./serving/deploy_vllm.sh <checkpoint_ordner>
# =============================================================================

set -e

CHECKPOINT_NAME=${1:-""}
if [ -z "$CHECKPOINT_NAME" ]; then
    echo "Fehler: Bitte Checkpoint-Ordner angeben."
    echo "Usage: ./serving/deploy_vllm.sh <checkpoint_ordner>"
    echo ""
    echo "Verfügbare Checkpoints:"
    ls data/final/checkpoints/ 2>/dev/null || echo "(keine)"
    exit 1
fi

ADAPTER_PATH="data/final/checkpoints/${CHECKPOINT_NAME}"
MERGED_PATH="data/final/checkpoints/${CHECKPOINT_NAME}_merged"
PORT=8000

echo "============================================================"
echo "  Text-to-SQL Modell Deployment"
echo "  Checkpoint: $CHECKPOINT_NAME"
echo "  Merged:     $MERGED_PATH"
echo "  Port:       $PORT"
echo "============================================================"

# ---------------------------------------------------------------------------
# Schritt 1: LoRA Adapter mergen (falls noch nicht geschehen)
# ---------------------------------------------------------------------------
if [ ! -d "$MERGED_PATH" ]; then
    echo ""
    echo "[1/2] Merge LoRA Adapter in Basismodell..."
    docker compose -f docker/docker-compose.yml run --rm training \
        python3 serving/merge_adapter.py \
        --adapter-path "$ADAPTER_PATH" \
        --output-path "$MERGED_PATH" \
        --config config/pipeline_config.yaml
    echo "✓ Merge abgeschlossen: $MERGED_PATH"
else
    echo "[1/2] Merged Modell bereits vorhanden – überspringe Merge."
fi

# ---------------------------------------------------------------------------
# Schritt 2: vLLM Server starten
#
# Wir nutzen das offizielle vLLM Docker Image – aber da vLLM kein
# offizielles SM_121 Image hat, nutzen wir das NGC Image + vLLM Installation.
#
# vLLM für SM_121 (GB10 Blackwell) muss aus dem Nightly gebaut werden.
# Wir nutzen den vLLM-nightly Index der SM_121 unterstützt.
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] Starte vLLM Server auf Port $PORT..."
echo "      (Erste Ausführung: vLLM wird installiert, dauert ~5 Min)"
echo "      Zum Beenden: Ctrl+C"
echo ""

docker compose -f docker/docker-compose.yml run --rm \
    -p ${PORT}:8000 \
    -e VLLM_USE_FLASHINFER_SAMPLER=0 \
    training bash -c "
        # vLLM für SM_121 installieren falls nicht vorhanden
        python3 -c 'import vllm' 2>/dev/null || {
            echo 'Installiere vLLM für SM_121 (GB10)...'
            pip install -q vllm \
                --extra-index-url https://wheels.vllm.ai/nightly \
                2>/dev/null || \
            pip install -q vllm 2>/dev/null
        }

        echo 'Starte vLLM API Server...'
        python3 -m vllm.entrypoints.openai.api_server \
            --model /app/${MERGED_PATH} \
            --served-model-name text2sql \
            --host 0.0.0.0 \
            --port 8000 \
            --dtype bfloat16 \
            --max-model-len 2048 \
            --gpu-memory-utilization 0.85 \
            --trust-remote-code
    "
