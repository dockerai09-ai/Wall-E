#!/usr/bin/env bash
# Synthesize the benchmark audio from cases.jsonl.
#
# macOS: uses the built-in `say` command with the Swedish voice (Alva) and
# afconvert to produce 16 kHz mono WAV. Linux: set TTS_CMD to a command that
# takes the text on stdin and writes a WAV to the path in $1, e.g.
#   TTS_CMD='piper --model sv_SE-nst-medium --output_file' ./synth.sh
#
#   VOICE=Alva RATE=175 ./synth.sh            # defaults
#   OUT=audio-fast RATE=230 ./synth.sh        # a faster-speech variant
#
# Writes <OUT>/<id>.wav and <OUT>/manifest.jsonl (cases + audio paths).
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
VOICE=${VOICE:-Alva}
RATE=${RATE:-175}
OUT=${OUT:-$DIR/audio}
mkdir -p "$OUT"

if [ -n "${TTS_CMD:-}" ]; then
  MODE=custom
elif command -v say >/dev/null 2>&1 && command -v afconvert >/dev/null 2>&1; then
  MODE=macos
else
  echo "synth.sh: need macOS 'say' + 'afconvert', or TTS_CMD (see header)" >&2
  exit 1
fi

n=0
: > "$OUT/manifest.jsonl"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  id=$(printf '%s' "$line" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
  text=$(printf '%s' "$line" | python3 -c 'import json,sys; print(json.load(sys.stdin)["reference"])')
  wav="$OUT/$id.wav"
  if [ ! -s "$wav" ]; then
    case $MODE in
      macos)
        say -v "$VOICE" -r "$RATE" -o "$OUT/$id.aiff" "$text"
        afconvert -f WAVE -d LEI16@16000 -c 1 "$OUT/$id.aiff" "$wav"
        rm -f "$OUT/$id.aiff"
        ;;
      custom)
        printf '%s' "$text" | $TTS_CMD "$wav"
        ;;
    esac
  fi
  printf '%s' "$line" | python3 -c "import json,sys; r=json.load(sys.stdin); r['audio']=sys.argv[1]; print(json.dumps(r, ensure_ascii=False))" "$wav" >> "$OUT/manifest.jsonl"
  n=$((n+1))
done < "$DIR/cases.jsonl"
echo "synthesized $n clips into $OUT (voice=$VOICE rate=$RATE) → $OUT/manifest.jsonl"
