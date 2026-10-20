#!/usr/bin/env bash
# Restores the binary assets that are not committed (fonts + temp score).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p fonts audio
cp /usr/share/fonts/opentype/inter/{InterDisplay-Black,InterDisplay-Bold,Inter-SemiBold,Inter-Medium}.otf fonts/
cp /usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Medium.ttf fonts/
tmp="$(mktemp -d)"
python3 -I scripts/gen-audio.py "$tmp/score.wav"
ffmpeg -y -loglevel error -i "$tmp/score.wav" -c:a aac -b:a 160k audio/score.m4a
rm -rf "$tmp"
