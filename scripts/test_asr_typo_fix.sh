#!/usr/bin/env bash
# Test post-Whisper typo fix (Groq / Gemini) without audio.
#
# Usage:
#   ./scripts/test_asr_typo_fix.sh
#   ./scripts/test_asr_typo_fix.sh 'ለስንዴ ማዳብሪያ ስንት መጠን ይሰጣል'
#   ASR_URL=http://127.0.0.1:8001 ASR_LLM_FIX_BACKEND=gemini ./scripts/test_asr_typo_fix.sh
#
# Force Gemini only (skip Groq): set ASR_LLM_FIX_BACKEND=gemini in .env and restart asr-service.
# Keys: GEMINI_API_KEY or GEMINI_API_KEYS in .env (same as RAG).

set -euo pipefail

ASR_URL="${ASR_URL:-http://127.0.0.1:8001}"
TEXT="${1:-ለስንዴ ማዳብሪያ ስንት መጠን ይሰጣል}"

echo "=== fix-status ==="
curl -sS "${ASR_URL}/fix-status" | python3 -m json.tool

echo ""
echo "=== postprocess-text (simulated bad ASR) ==="
echo "input: $TEXT"
curl -sS -X POST "${ASR_URL}/postprocess-text" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "import json,sys; print(json.dumps({'text': sys.argv[1]}))" "$TEXT")" \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
for k in ('input', 'domain_corrected_transcript', 'semantic_corrected_transcript',
          'transcript_fix_backend', 'final_transcript'):
    print(f'{k}: {d.get(k)!r}')
"

echo ""
echo "If transcript_fix_backend is null, set GEMINI_API_KEY in .env, ASR_HOSTED_LLM_FIX=auto, then:"
echo "  docker compose up -d --build asr-service"
