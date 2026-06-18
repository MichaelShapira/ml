# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import sys
import glob
import json
import time
import ctypes
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- preload the cuDNN / cuBLAS libs that ship with the DLC's torch wheels, so
# CTranslate2 can find them on the GPU (avoids "libcudnn_ops.so.9 not found"). ---
def _preload_cuda_libs():
    roots = []
    try:
        import site
        roots += list(site.getsitepackages())
    except Exception:
        pass
    roots += [p for p in sys.path if p.endswith("site-packages")]
    patterns = ["nvidia/cublas/lib/libcublas*.so*", "nvidia/cudnn/lib/libcudnn*.so*"]
    seen = set()
    for r in roots:
        for pat in patterns:
            for so in sorted(glob.glob(os.path.join(r, pat))):
                if so in seen:
                    continue
                seen.add(so)
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


_preload_cuda_libs()

import ctranslate2
from faster_whisper import WhisperModel

device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
compute_type = os.environ.get("compute_type") or ("float16" if device == "cuda" else "int8")
# ivrit-ai model is Yiddish; force the language so we skip (sometimes wrong) detection.
language = os.environ.get("language", "yi") or None
beam_size = int(os.environ.get("beam_size", "5"))
vad_filter = os.environ.get("vad_filter", "true").lower() == "true"


def model_fn(model_dir):
    logger.info("Loading faster-whisper from %s (device=%s, compute_type=%s)",
                model_dir, device, compute_type)
    return WhisperModel(model_dir, device=device, compute_type=compute_type)


def transform_fn(model, request_body, request_content_type, response_content_type="application/json"):
    start = time.time()
    tfile = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
    try:
        tfile.write(request_body)
        tfile.flush()
        tfile.close()

        logger.info("Transcribing (language=%s, vad_filter=%s, beam=%d) ...",
                    language, vad_filter, beam_size)
        segments, info = model.transcribe(
            tfile.name,
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=vad_filter,
        )

        words, seg_list, full = [], [], []
        for seg in segments:                      # generator -> materialize
            seg_list.append({"start": seg.start, "end": seg.end, "text": seg.text})
            full.append(seg.text)
            for w in (seg.words or []):
                words.append({"timestamp": [w.start, w.end], "text": w.word})
    finally:
        try:
            os.remove(tfile.name)
        except OSError:
            pass

    logger.info("Done in %.1fs (%d words, %d segments)",
                time.time() - start, len(words), len(seg_list))

    # `chunks` holds word-level tokens (same shape consumers already expect);
    # `segments` keeps faster-whisper's natural sentence-ish segments too.
    result = {"text": "".join(full), "segments": seg_list, "chunks": words}
    return json.dumps(result, ensure_ascii=False), response_content_type
