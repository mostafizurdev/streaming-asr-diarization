#!/usr/bin/env bash
# Download the pretrained models (all from the sherpa-onnx GitHub releases) into ./models.
#   scripts/download_models.sh          # ASR + speaker model (~340 MB)
#   scripts/download_models.sh --bench  # also the Piper TTS voice used to build test conversations
set -euo pipefail

BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"
DIR="${MODELS_DIR:-models}"
mkdir -p "$DIR"
cd "$DIR"

fetch_tar() {  # $1 = release tag, $2 = archive name without extension
  if [ ! -d "$2" ]; then
    echo "downloading $2"
    curl -fSL --retry 3 -o "$2.tar.bz2" "$BASE/$1/$2.tar.bz2"
    tar xjf "$2.tar.bz2" && rm "$2.tar.bz2"
  fi
}

# Streaming Zipformer transducer for English (icefall recipe, exported by sherpa-onnx).
fetch_tar asr-models sherpa-onnx-streaming-zipformer-en-2023-06-26
# Keep only the int8 weights to save space.
rm -f sherpa-onnx-streaming-zipformer-en-2023-06-26/{encoder,decoder,joiner}-epoch-99-avg-1-chunk-16-left-128.onnx

# NVIDIA NeMo TitaNet-small speaker embedding model, exported to ONNX.
if [ ! -f nemo_en_titanet_small.onnx ]; then
  echo "downloading nemo_en_titanet_small.onnx"
  curl -fSL --retry 3 -o nemo_en_titanet_small.onnx "$BASE/speaker-recongition-models/nemo_en_titanet_small.onnx"
fi

if [ "${1:-}" = "--bench" ]; then
  # Multi-speaker Piper voice (904 LibriTTS-R speakers) used only to synthesise test conversations.
  fetch_tar tts-models vits-piper-en_US-libritts_r-medium
fi
echo "models ready in $DIR"
