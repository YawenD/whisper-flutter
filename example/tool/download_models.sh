#!/usr/bin/env bash
# Download the models used by the example app that are too large for git.
#   - ggml-small-q5_1.bin : whisper.cpp, default on iOS (lib/stream.dart)
#   - ggml-base-q5_1.bin  : whisper.cpp, Android fallback and benchmarks
#   - sherpa-onnx Whisper base (fp32): default engine on Android
# Pass --int8 to also fetch the sherpa-onnx int8 exports (lib/bench.dart only).
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

echo "Models ready."
