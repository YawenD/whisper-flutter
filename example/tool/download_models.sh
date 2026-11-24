#!/usr/bin/env bash
# Download the models used by the example app that are too large for git.
#   - ggml-small-q5_1.bin : whisper.cpp, default on iOS (lib/stream.dart)
#   - ggml-base-q5_1.bin  : whisper.cpp, Android fallback and benchmarks
#   - NVIDIA Canary-180M-Flash (sherpa-onnx int8): default engine on Android
#   - NVIDIA FastConformer transducer en/de/es/fr (sherpa-onnx int8)
#   - sherpa-onnx Whisper base (fp32): comparison engine
# Pass --int8 to also fetch the Whisper base int8 exports (lib/bench.dart).
# NVIDIA models are CC-BY-4.0: credit NVIDIA in the app.
set -euo pipefail
cd "$(dirname "$0")/.."

HF=https://huggingface.co/ggerganov/whisper.cpp/resolve/main
SHERPA=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models

mkdir -p assets/models assets/sherpa/whisper-base

for model in ggml-small-q5_1.bin ggml-base-q5_1.bin; do
  if [ ! -f "assets/models/$model" ]; then
    echo "Downloading $model..."
    curl -L --fail -o "assets/models/$model.tmp" "$HF/$model"
    mv "assets/models/$model.tmp" "assets/models/$model"
  fi
done

dest=assets/sherpa/whisper-base
if [ ! -f "$dest/base-encoder.onnx" ] || { [ "${1:-}" = "--int8" ] && [ ! -f "$dest/base-encoder.int8.onnx" ]; }; then
  echo "Downloading sherpa-onnx Whisper base..."
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  curl -L --fail -o "$tmp/whisper-base.tar.bz2" "$SHERPA/sherpa-onnx-whisper-base.tar.bz2"
  tar xjf "$tmp/whisper-base.tar.bz2" -C "$tmp"
  src="$tmp/sherpa-onnx-whisper-base"
  cp "$src/base-encoder.onnx" "$src/base-decoder.onnx" "$src/base-tokens.txt" "$dest/"
  if [ "${1:-}" = "--int8" ]; then
    cp "$src/base-encoder.int8.onnx" "$src/base-decoder.int8.onnx" "$dest/"
  fi
fi

# NVIDIA NeMo models: archive name -> destination folder.
fetch_nemo() {
  local archive=$1 dest=$2
  if [ -f "$dest/tokens.txt" ]; then return; fi
  echo "Downloading $archive..."
  mkdir -p "$dest"
  local tmp
  tmp=$(mktemp -d)
  curl -L --fail -o "$tmp/model.tar.bz2" "$SHERPA/$archive.tar.bz2"
  tar xjf "$tmp/model.tar.bz2" -C "$tmp"
  cp "$tmp/$archive"/*.onnx "$tmp/$archive/tokens.txt" "$dest/"
  rm -rf "$tmp"
}
fetch_nemo sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8 assets/sherpa/canary-180m
fetch_nemo sherpa-onnx-nemo-fast-conformer-transducer-en-de-es-fr-14288-int8 assets/sherpa/fc-transducer

echo "Models ready."
