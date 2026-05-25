#!/usr/bin/env bash
# Test ASR service in isolation (upload WAV → JSON transcript).
#
# Usage:
#   ./scripts/test_asr_upload.sh /path/to/audio.wav
#   ASR_URL=http://127.0.0.1:8001 ./scripts/test_asr_upload.sh sample.wav
#
# Docker Compose default host port for asr-service is 8001.
# Expects: 16 kHz mono PCM WAV works best (same as VAD utterances).

set -euo pipefail

WAV="${1:-}"
ASR_URL="${ASR_URL:-http://127.0.0.1:8001}"

if [[ -z "$WAV" ]] || [[ ! -f "$WAV" ]]; then
  echo "Usage: ASR_URL=http://host:port $0 <audio.wav>" >&2
  exit 1
fi

echo "POST $ASR_URL/transcribe  file=@$(basename "$WAV")"
curl -sS -X POST "${ASR_URL}/transcribe" \
  -H "Accept: application/json" \
  -F "file=@${WAV}" \
  | python3 -m json.tool

echo ""
echo "Tip: compare raw_transcript, semantic_corrected_transcript, transcript_fix_backend, final_transcript."
echo "Text-only typo test (no Whisper): ./scripts/test_asr_typo_fix.sh"
echo "Tune ASR in asr_service/config.py and postprocess.py if text is off."
