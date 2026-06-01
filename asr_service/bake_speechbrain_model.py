#!/usr/bin/env python3
"""
Pre-download SpeechBrain Amharic ASR weights into the Docker image (run once at build).

Env:
  ASR_SPEECHBRAIN_SOURCE  (default: speechbrain/asr-wav2vec2-dvoice-amharic)
  ASR_SPEECHBRAIN_SAVEDIR (default: /opt/asr-models/speechbrain-wav2vec2-dvoice-amharic)
  HF_HOME                 (default: /opt/asr-models/hf-cache)
  HF_TOKEN                (optional, faster Hub downloads)
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path


def main() -> int:
    source = os.getenv(
        "ASR_SPEECHBRAIN_SOURCE", "speechbrain/asr-wav2vec2-dvoice-amharic"
    ).strip()
    savedir = Path(
        os.getenv(
            "ASR_SPEECHBRAIN_SAVEDIR",
            "/opt/asr-models/speechbrain-wav2vec2-dvoice-amharic",
        )
    )
    hf_home = Path(os.getenv("HF_HOME", "/opt/asr-models/hf-cache"))
    savedir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    print(f"[bake] source={source}", flush=True)
    print(f"[bake] savedir={savedir}", flush=True)
    print(f"[bake] HF_HOME={hf_home}", flush=True)

    try:
        from speechbrain.inference.ASR import EncoderASR
    except ImportError:
        from speechbrain.pretrained import EncoderASR

    # Downloads hyperparams + wav2vec2-large-xlsr-53 (~1.2GB) into savedir/cache.
    model = EncoderASR.from_hparams(
        source=source,
        savedir=str(savedir),
        run_opts={"device": "cpu"},
    )
    del model
    gc.collect()

    files = list(savedir.rglob("*"))
    print(f"[bake] done — {len(files)} files under {savedir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
