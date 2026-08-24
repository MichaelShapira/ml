#!/usr/bin/env python3
"""
SageMaker inference shim for MiniMax-H3 served by SGLang Diffusion.

Why a shim exists
-----------------
SGLang Diffusion already speaks an OpenAI-style *asynchronous* video API
(``POST /v1/videos`` -> job id -> poll -> ``GET /v1/videos/{id}/content``).
SageMaker speaks a *synchronous* ``POST /invocations`` with a ``GET /ping``
health gate. This process:

  1. launches one SGLang backend per H3 checkpoint partition, each pinned to
     its own slice of GPUs (``CUDA_VISIBLE_DEVICES``),
  2. gates ``/ping`` on those backends actually being able to generate,
  3. turns one ``/invocations`` call into submit -> poll -> download,
  4. materialises ``conditions[].uri`` from ``s3://`` / ``data:`` inputs into
     container-local ``file://`` paths that SGLang can read,
  5. writes the finished MP4s to S3 and returns JSON metadata (or inlines
     base64 when no output prefix is configured).

Routing: ``t2va`` and ``fl2va`` are served by the FL2VA partition, ``ref2va``
by the Ref2VA partition. That is a property of the released checkpoints, not a
choice made here.

Everything is stdlib + boto3 on purpose. The SGLang image pins a delicate
torch / diffusers / transformers set and adding a web framework here is a good
way to break it.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
HUB_REPO_ID = os.environ.get("H3_HUB_REPO_ID", "MiniMaxAI/MiniMax-H3")

# Which checkpoint partitions to serve, in GPU order.
VARIANTS = [v.strip() for v in os.environ.get("H3_VARIANTS", "fl2va,ref2va").split(",") if v.strip()]

GPUS_PER_BACKEND = int(os.environ.get("H3_GPUS_PER_BACKEND", "4"))
TP_SIZE = int(os.environ.get("H3_TP_SIZE", "4"))
ULYSSES_DEGREE = int(os.environ.get("H3_ULYSSES_DEGREE", "1"))
PERFORMANCE_MODE = os.environ.get("H3_PERFORMANCE_MODE", "speed")
EXTRA_ARGS = os.environ.get("H3_EXTRA_ARGS", "").split()
WARMUP_RESOLUTIONS = os.environ.get("H3_WARMUP_RESOLUTIONS", "").strip()

BASE_PORT = int(os.environ.get("H3_BASE_PORT", "30010"))
# Backends' HTTP ports must be SPACED, not consecutive: SGLang additionally
# binds a ZMQ job broker on (http port + 1). With a stride of 1, backend 0's
# broker takes exactly the port backend 1 was told to serve HTTP on; SGLang
# then silently "settles" backend 1's HTTP server onto a free port (+42),
# the shim keeps probing the original port - now a ZMQ socket - and backend 1
# never reports healthy. Symptom in the logs: backend 0 spams
# "Error in ZMQ Broker: Resource temporarily unavailable" from the moment
# backend 1 launches, and boot times out after H3_BACKEND_BOOT_TIMEOUT_S.
PORT_STRIDE = int(os.environ.get("H3_PORT_STRIDE", "10"))
if PORT_STRIDE < 2:
    raise SystemExit(f"H3_PORT_STRIDE must be >= 2 (got {PORT_STRIDE}): port+1 is SGLang's ZMQ broker")
BIND_PORT = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080"))

# Gate /ping on a real (tiny) generation rather than just an open socket.
WARMUP_ON_START = os.environ.get("H3_WARMUP_ON_START", "1") not in ("0", "false", "False")
WARMUP_STEPS = int(os.environ.get("H3_WARMUP_STEPS", "4"))
# 768 is not a preference: SGLang rejects any other short_edge for minimax_h3
# ("target.short_edge must be 768 for minimax_h3"), so a smaller/cheaper
# warmup frame is not possible. Cost is controlled by WARMUP_STEPS instead.
WARMUP_SHORT_EDGE = int(os.environ.get("H3_WARMUP_SHORT_EDGE", "768"))

BACKEND_BOOT_TIMEOUT_S = int(os.environ.get("H3_BACKEND_BOOT_TIMEOUT_S", "3000"))
JOB_TIMEOUT_S = int(os.environ.get("H3_JOB_TIMEOUT_S", "3300"))
JOB_POLL_INTERVAL_S = float(os.environ.get("H3_JOB_POLL_INTERVAL_S", "2.0"))

OUTPUT_S3_PREFIX = os.environ.get("H3_OUTPUT_S3_PREFIX", "").strip()
SCRATCH_DIR = Path(os.environ.get("H3_SCRATCH_DIR", "/tmp/h3"))

# Model constraints, from the model card + the SGLang cookbook.
MIN_DURATION_S = 4.0
MAX_DURATION_S = 15.0
MAX_OUTPUTS_PER_PROMPT = 10
REF_LIMITS = {"image": 9, "video": 3, "audio": 3}
REF_TOTAL_LIMIT = 12
TASKS = ("t2va", "fl2va", "ref2va")
VARIANT_FOR_TASK = {"t2va": "fl2va", "fl2va": "fl2va", "ref2va": "ref2va"}

# Fields forwarded verbatim to SGLang's /v1/videos when present.
PASSTHROUGH_FIELDS = (
    "seed",
    "num_inference_steps",
    "flow_shift",
    "audio_flow_shift",
    "num_outputs_per_prompt",
    "n",
    "quality",
    "output_quality",
    "seconds",
)
# Deliberately absent: negative_prompt / guidance_scale. The released H3
# checkpoints are CFG-distilled, so guidance is baked into the weights and there
# is exactly one forward pass per step - forwarding those fields would suggest
# they do something.

_START_TS = time.time()


def log(msg: str) -> None:
    print(f"[shim +{time.time() - _START_TS:7.1f}s] {msg}", flush=True)


class RequestError(Exception):
    """A 4xx: the caller's payload is wrong."""


# --------------------------------------------------------------------------- #
# weight resolution
# --------------------------------------------------------------------------- #


def _find_repo_root(model_dir: str) -> Path | None:
    """Locate the staged MiniMax-H3 repository tree under /opt/ml/model.

    Accepts either the repo laid down directly in ``model_dir`` or nested one
    level (``model_dir/MiniMax-H3``), which is what ``hf download --local-dir``
    plus ``aws s3 sync`` normally produces.
    """
    root = Path(model_dir)
    if not root.is_dir():
        return None
    candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir())
    for cand in candidates:
        if (cand / "FL2VA").is_dir() or (cand / "Ref2VA").is_dir():
            return cand
    return None


def _link_into_hf_cache(repo_root: Path) -> str:
    """Present a staged repo tree to huggingface_hub as a warm cache entry.

    SGLang owns the checkpoint-directory mapping and resolves ``--model-path``
    as a Hub id, so the robust way to serve local weights is to make the Hub
    believe they are already downloaded. Snapshot entries are symlinks, so this
    costs no extra disk and no copy of 288 GB.

    Returns the repo id to pass as ``--model-path``.
    """
    hf_home = Path(os.environ.setdefault("HF_HOME", "/tmp/h3-hf"))
    marker = repo_root / "_h3_stage.json"
    revision = "main"
    if marker.is_file():
        try:
            revision = json.loads(marker.read_text()).get("revision") or "main"
        except (OSError, ValueError):
            pass

    cache_root = hf_home / "hub" / ("models--" + HUB_REPO_ID.replace("/", "--"))
    snapshot = cache_root / "snapshots" / revision
    if snapshot.exists():
        shutil.rmtree(snapshot, ignore_errors=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    (cache_root / "refs").mkdir(parents=True, exist_ok=True)
    (cache_root / "refs" / "main").write_text(revision)

    linked = 0
    for entry in repo_root.iterdir():
        if entry.name.startswith("_h3_"):
            continue
        target = snapshot / entry.name
        if not target.exists():
            target.symlink_to(entry)
            linked += 1

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    log(f"HF cache primed: {snapshot} ({linked} top-level links, revision={revision}), HF_HUB_OFFLINE=1")
    return HUB_REPO_ID


def resolve_model_path() -> str:
    """Decide what to hand ``sglang serve --model-path``."""
    override = os.environ.get("H3_MODEL_PATH", "").strip()
    if override:
        log(f"H3_MODEL_PATH override: {override}")
        return override

    repo_root = _find_repo_root(MODEL_DIR)
    if repo_root is None:
        log(
            f"no staged weights under {MODEL_DIR} -> falling back to the Hub id "
            f"{HUB_REPO_ID}. Expect a long first-boot download."
        )
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        return HUB_REPO_ID

    log(f"staged weights found at {repo_root}")
    present = [d.name for d in sorted(repo_root.iterdir()) if d.is_dir()]
    log(f"  partitions/components present: {present}")
    for variant in VARIANTS:
        expected = {"fl2va": "FL2VA", "ref2va": "Ref2VA"}[variant]
        if not (repo_root / expected).is_dir():
            log(f"  WARNING: variant '{variant}' requested but {expected}/ is missing from the staged tree")
    return _link_into_hf_cache(repo_root)


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #


class Backend:
    def __init__(self, variant: str, index: int, model_path: str) -> None:
        self.variant = variant
        self.index = index
        self.model_path = model_path
        self.port = BASE_PORT + index * PORT_STRIDE
        self.gpus = list(range(index * GPUS_PER_BACKEND, (index + 1) * GPUS_PER_BACKEND))
        self.proc: subprocess.Popen[str] | None = None
        self.ready = False
        self.error: str | None = None
        self.health_path: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            self.model_path,
            "--model-variant",
            self.variant,
            "--num-gpus",
            str(GPUS_PER_BACKEND),
            "--performance-mode",
            PERFORMANCE_MODE,
            # Bound to loopback: the only way in is this shim's /invocations.
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        # Emit both degrees explicitly even when one is 1: the cookbook's
        # recipes are written that way, and leaning on defaults here is how you
        # end up serving a topology you did not choose.
        cmd += ["--tp-size", str(TP_SIZE), "--ulysses-degree", str(ULYSSES_DEGREE)]
        if WARMUP_RESOLUTIONS:
            cmd += ["--warmup-resolutions", WARMUP_RESOLUTIONS]
        cmd += EXTRA_ARGS
        return cmd

    def start(self) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpus)
        # Two independent 4-GPU process groups on one host: keep their
        # torch.distributed rendezvous ports apart.
        env["MASTER_PORT"] = str(29500 + self.index * 100)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        cmd = self.command()
        log(f"[{self.variant}] launching on GPUs {env['CUDA_VISIBLE_DEVICES']} port {self.port}")
        log(f"[{self.variant}] $ {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._pump_logs, daemon=True).start()

    def _pump_logs(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            print(f"[{self.variant}] {line.rstrip()}", flush=True)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def probe_health(self) -> bool:
        """True once the backend's HTTP app answers.

        SGLang's diffusion server has moved health paths around between
        revisions, so try the known candidates and remember which one worked
        instead of hard-coding one and silently never going ready.
        """
        paths = [self.health_path] if self.health_path else ["/health", "/v1/models", "/get_model_info", "/health_generate"]
        for path in paths:
            if not path:
                continue
            try:
                with urllib.request.urlopen(self.base_url + path, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        if self.health_path != path:
                            log(f"[{self.variant}] health endpoint = {path}")
                            self.health_path = path
                        return True
            except (urllib.error.URLError, OSError, TimeoutError):
                continue
        return False

    def warmup(self) -> None:
        """Run one tiny generation so InService means 'can actually generate'."""
        payload = {
            "task": "t2va" if self.variant == "fl2va" else "ref2va",
            "prompt": "A single candle flame flickering in a dark room, quiet room tone.",
            "conditions": [],
            "target": {
                "short_edge": WARMUP_SHORT_EDGE,
                "aspect_ratio": "16:9",
                "duration_seconds": MIN_DURATION_S,
            },
            "num_inference_steps": WARMUP_STEPS,
            "seed": 0,
        }
        if self.variant == "ref2va":
            # Ref2VA needs at least one reference; a solid-colour PNG is enough
            # to exercise the packing path without depending on the network.
            png = SCRATCH_DIR / "warmup.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            png.write_bytes(base64.b64decode(_TINY_PNG_B64))
            payload["prompt"] = (
                "subject_definitions:\n<Picture 1> is a plain grey frame.\n\n"
                "summary:\n[reference generation] A slow push in on a plain grey wall.\n\n"
                "detailed_description:\n[Shot 1] Static shot of a plain grey wall, "
                "the camera pushes in with small amplitude at slow speed.\n\n"
                "overall_soundscape:\nQuiet room tone throughout.\n\n"
                "non_diegetic_music:\nN/A"
            )
            payload["conditions"] = [{"type": "image", "uri": f"file://{png}", "role": "reference"}]
        t0 = time.time()
        log(f"[{self.variant}] warmup generation starting ({WARMUP_STEPS} steps, short edge {WARMUP_SHORT_EDGE})")
        job = submit_and_wait(self, payload, timeout_s=BACKEND_BOOT_TIMEOUT_S)
        fetch_variant_bytes(self, job["id"], 0)
        log(f"[{self.variant}] warmup complete in {time.time() - t0:.1f}s")


# 1x1 grey PNG.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

BACKENDS: dict[str, Backend] = {}
_READY = threading.Event()
_BOOT_ERROR: list[str] = []


def _log_host_facts() -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        log("GPUs:\n  " + "\n  ".join(out.stdout.strip().splitlines() or ["(none reported)"]))
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        log(f"nvidia-smi unavailable: {exc}")
    for path in (MODEL_DIR, "/tmp"):
        try:
            usage = shutil.disk_usage(path)
            log(f"disk {path}: {usage.used / 1e9:.0f}/{usage.total / 1e9:.0f} GB used")
        except OSError:
            pass


def boot() -> None:
    """Start every backend, then flip readiness once all of them can generate."""
    try:
        _log_host_facts()
        model_path = resolve_model_path()
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

        for i, variant in enumerate(VARIANTS):
            backend = Backend(variant, i, model_path)
            BACKENDS[variant] = backend
            backend.start()
            # Stagger: two backends racing for a free rendezvous port during
            # torch.distributed init is a real (and confusing) failure mode.
            _wait_for_health(backend)

        if WARMUP_ON_START:
            for backend in BACKENDS.values():
                backend.warmup()

        for backend in BACKENDS.values():
            backend.ready = True
        _READY.set()
        log(f"READY - serving variants {list(BACKENDS)} (ports {[b.port for b in BACKENDS.values()]})")
    except Exception as exc:  # noqa: BLE001 - must surface, not vanish
        _BOOT_ERROR.append(f"{type(exc).__name__}: {exc}")
        log("BOOT FAILED:\n" + traceback.format_exc())


def _wait_for_health(backend: Backend) -> None:
    deadline = time.time() + BACKEND_BOOT_TIMEOUT_S
    last_note = 0.0
    while time.time() < deadline:
        if not backend.alive():
            code = backend.proc.returncode if backend.proc else "?"
            raise RuntimeError(
                f"backend '{backend.variant}' exited with code {code} before becoming healthy "
                f"- see the [{backend.variant}] lines above for the real traceback"
            )
        if backend.probe_health():
            log(f"[{backend.variant}] HTTP up after {time.time() - _START_TS:.0f}s")
            return
        if time.time() - last_note > 60:
            last_note = time.time()
            log(f"[{backend.variant}] still loading ({time.time() - _START_TS:.0f}s elapsed)")
        time.sleep(3)
    raise TimeoutError(f"backend '{backend.variant}' did not become healthy within {BACKEND_BOOT_TIMEOUT_S}s")


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"backend {exc.code} on {url}: {detail}") from exc
    return json.loads(body.decode("utf-8"))


def _http_bytes(url: str, timeout: float = 600.0) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"backend {exc.code} on {url}: {detail}") from exc


def submit_and_wait(backend: Backend, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    job = _http_json(f"{backend.base_url}/v1/videos", payload)
    job_id = job.get("id")
    if not job_id:
        raise RuntimeError(f"backend did not return a job id: {json.dumps(job)[:500]}")
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        state = _http_json(f"{backend.base_url}/v1/videos/{job_id}")
        status = state.get("status")
        if status != last_status:
            log(f"[{backend.variant}] job {job_id} status={status}")
            last_status = status
        if status in ("completed", "succeeded"):
            state.setdefault("id", job_id)
            return state
        if status in ("failed", "cancelled", "error"):
            raise RuntimeError(f"generation {status}: {json.dumps(state)[:2000]}")
        if not backend.alive():
            raise RuntimeError(f"backend '{backend.variant}' died while job {job_id} was running")
        time.sleep(JOB_POLL_INTERVAL_S)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s}s")


def fetch_variant_bytes(backend: Backend, job_id: str, variant_index: int) -> bytes:
    url = f"{backend.base_url}/v1/videos/{job_id}/content"
    if variant_index:
        url += f"?variant={variant_index}"
    return _http_bytes(url)


# --------------------------------------------------------------------------- #
# S3
# --------------------------------------------------------------------------- #

_s3_lock = threading.Lock()
_s3_client = None


def s3():
    global _s3_client
    with _s3_lock:
        if _s3_client is None:
            import boto3  # imported lazily so a missing boto3 cannot break boot

            _s3_client = boto3.client("s3")
        return _s3_client


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise RequestError(f"not an s3 uri: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


# --------------------------------------------------------------------------- #
# request handling
# --------------------------------------------------------------------------- #

_EXT_FOR_TYPE = {"image": ".png", "video": ".mp4", "audio": ".wav"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _ext_from(uri: str, cond_type: str) -> str:
    guess = Path(urllib.parse.urlparse(uri).path).suffix.lower()
    if guess and len(guess) <= 6:
        return guess
    return _EXT_FOR_TYPE.get(cond_type, ".bin")


def _materialize_condition(cond: dict[str, Any], workdir: Path, idx: int) -> dict[str, Any]:
    """Turn one condition into something SGLang can open.

    ``s3://`` and inline base64 are downloaded to the container's scratch dir
    and rewritten to ``file://``. ``http(s)://`` and ``file://`` are passed
    through - SGLang fetches those itself.
    """
    if not isinstance(cond, dict):
        raise RequestError(f"conditions[{idx}] must be an object, got {type(cond).__name__}")
    out = dict(cond)
    cond_type = out.get("type")
    if cond_type not in REF_LIMITS:
        raise RequestError(f"conditions[{idx}].type must be one of {sorted(REF_LIMITS)}, got {cond_type!r}")

    inline = out.pop("data", None) or out.pop("base64", None)
    uri = out.get("uri") or out.get("url")
    out.pop("url", None)

    if inline:
        ext = out.pop("ext", None) or _EXT_FOR_TYPE.get(cond_type, ".bin")
        path = workdir / f"cond{idx}{ext}"
        path.write_bytes(base64.b64decode(inline))
        out["uri"] = f"file://{path}"
        return out

    if not uri:
        raise RequestError(f"conditions[{idx}] needs 'uri' (or inline 'data' base64)")
    if not isinstance(uri, str):
        raise RequestError(f"conditions[{idx}].uri must be a string")

    if uri.startswith("data:"):
        header, _, b64 = uri.partition(",")
        mime = header[5:].split(";")[0]
        ext = mimetypes.guess_extension(mime) or _EXT_FOR_TYPE.get(cond_type, ".bin")
        path = workdir / f"cond{idx}{ext}"
        path.write_bytes(base64.b64decode(b64))
        out["uri"] = f"file://{path}"
        return out

    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        name = _SAFE_NAME.sub("_", Path(key).name) or f"cond{idx}{_ext_from(uri, cond_type)}"
        path = workdir / f"cond{idx}-{name}"
        s3().download_file(bucket, key, str(path))
        out["uri"] = f"file://{path}"
        return out

    if uri.startswith(("http://", "https://", "file://")):
        return out

    local = Path(uri)
    if local.is_file():
        out["uri"] = f"file://{local.resolve()}"
        return out

    raise RequestError(
        f"conditions[{idx}].uri {uri!r} is not an s3://, data:, http(s):// or file:// reference"
    )


def _validate_conditions(task: str, conditions: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for cond in conditions:
        counts[cond["type"]] = counts.get(cond["type"], 0) + 1

    if task == "t2va":
        if conditions:
            raise RequestError("task 't2va' takes no conditions; use 'fl2va' for keyframes")
        return

    if task == "fl2va":
        if not conditions:
            raise RequestError("task 'fl2va' needs 1 or 2 keyframe images (use 't2va' for text only)")
        if set(counts) != {"image"} or counts.get("image", 0) > 2:
            raise RequestError("task 'fl2va' takes 1 or 2 image conditions with role 'keyframe'")
        for i, cond in enumerate(conditions):
            if cond.get("role") not in (None, "keyframe"):
                raise RequestError(f"conditions[{i}].role must be 'keyframe' for task 'fl2va'")
        return

    # ref2va
    if not conditions:
        raise RequestError("task 'ref2va' needs at least one image, video or audio reference")
    for kind, limit in REF_LIMITS.items():
        if counts.get(kind, 0) > limit:
            raise RequestError(f"task 'ref2va' accepts at most {limit} {kind} references, got {counts[kind]}")
    if len(conditions) > REF_TOTAL_LIMIT:
        raise RequestError(f"task 'ref2va' accepts at most {REF_TOTAL_LIMIT} references, got {len(conditions)}")
    if counts.get("audio", 0) and not (counts.get("image", 0) or counts.get("video", 0)):
        raise RequestError("audio references must be accompanied by an image or video reference")


def build_backend_payload(body: dict[str, Any], workdir: Path) -> tuple[Backend, dict[str, Any], int]:
    task = body.get("task", "t2va")
    if task not in TASKS:
        raise RequestError(f"task must be one of {TASKS}, got {task!r}")
    backend = BACKENDS.get(VARIANT_FOR_TASK[task])
    if backend is None:
        raise RequestError(
            f"task {task!r} needs the '{VARIANT_FOR_TASK[task]}' checkpoint partition, "
            f"but this endpoint only serves {list(BACKENDS)} (see H3_VARIANTS)"
        )

    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RequestError("'prompt' is required and must be a non-empty string")

    target = dict(body.get("target") or {})
    duration = float(target.get("duration_seconds", body.get("seconds", 5)))
    if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        raise RequestError(
            f"target.duration_seconds must be between {MIN_DURATION_S} and {MAX_DURATION_S}, got {duration}"
        )
    target["duration_seconds"] = duration
    target.setdefault("short_edge", 768)
    target.setdefault("aspect_ratio", "auto" if task != "t2va" else "16:9")

    conditions = list(body.get("conditions") or [])
    materialized = [_materialize_condition(c, workdir, i) for i, c in enumerate(conditions)]
    _validate_conditions(task, materialized)

    n_outputs = int(body.get("num_outputs_per_prompt", body.get("n", 1)))
    if not (1 <= n_outputs <= MAX_OUTPUTS_PER_PROMPT):
        raise RequestError(f"num_outputs_per_prompt must be 1..{MAX_OUTPUTS_PER_PROMPT}, got {n_outputs}")

    payload: dict[str, Any] = {
        "task": task,
        "prompt": prompt,
        "conditions": materialized,
        "target": target,
        "num_outputs_per_prompt": n_outputs,
    }
    for field in PASSTHROUGH_FIELDS:
        if field in body and field not in payload:
            payload[field] = body[field]
    payload.pop("n", None)
    payload.setdefault("seconds", duration)
    extra = body.get("extra_body")
    if isinstance(extra, dict):
        payload.update(extra)
    return backend, payload, n_outputs


def _ffmpeg(*args: str) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        raise RuntimeError("ffmpeg failed: " + " | ".join(tail))


def _probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except (TypeError, ValueError):
        return 0.0


ALL_ARTIFACTS = {"first_frame": True, "last_frame": True, "audio": True}


def extract_artifacts(
    video_bytes: bytes,
    workdir: Path,
    want: dict[str, Any] | bool,
    bucket: str,
    key_prefix: str,
    variant: int,
) -> dict[str, str]:
    """Derive the first frame, last frame and soundtrack, and put them in S3.

    Done here rather than client side for one reason: the bytes are already in
    this process and ffmpeg is already in this image, so chaining a generated clip
    into a keyframe or reference request costs one S3 PUT instead of a full
    round-trip download and re-upload. Failures are reported per artifact and
    never fail the generation - the video is the deliverable.
    """
    # `extract: true` is as valid on the wire as an explicit dict, so normalise
    # here rather than trusting every caller to have done it.
    if not isinstance(want, dict):
        want = dict(ALL_ARTIFACTS) if want else {}
    if not want:
        return {}

    src = workdir / f"out{variant}.mp4"
    src.write_bytes(video_bytes)
    artifacts: dict[str, str] = {}
    duration = _probe_duration(src)

    jobs: list[tuple[str, str, list[str]]] = []
    if want.get("first_frame", True):
        jobs.append(("first_frame", f"first-{variant}.png",
                     ["-i", str(src), "-ss", "0", "-frames:v", "1"]))
    if want.get("last_frame", True) and duration > 0:
        # Back off a hair from the reported duration: seeking exactly to it can
        # land past the last decodable frame and write an empty file.
        jobs.append(("last_frame", f"last-{variant}.png",
                     ["-i", str(src), "-ss", f"{max(duration - 0.05, 0):.3f}", "-frames:v", "1"]))
    if want.get("audio", True):
        # H3's native rate: 32 kHz stereo.
        jobs.append(("audio", f"audio-{variant}.wav",
                     ["-i", str(src), "-vn", "-ac", "2", "-ar", "32000", "-c:a", "pcm_s16le"]))

    for name, filename, args in jobs:
        dest = workdir / filename
        try:
            _ffmpeg(*args, str(dest))
            if not dest.is_file() or dest.stat().st_size == 0:
                raise RuntimeError("produced an empty file")
            content_type = "image/png" if filename.endswith(".png") else "audio/wav"
            key = f"{key_prefix}/{filename}"
            s3().put_object(
                Bucket=bucket, Key=key, Body=dest.read_bytes(), ContentType=content_type
            )
            artifacts[name] = f"s3://{bucket}/{key}"
        except Exception as exc:  # noqa: BLE001 - never fail the generation over this
            log(f"extract {name} failed: {type(exc).__name__}: {exc}")
    if artifacts:
        log(f"extracted {', '.join(sorted(artifacts))}")
    return artifacts


def handle_invocation(body: dict[str, Any]) -> dict[str, Any]:
    request_id = body.get("request_id") or uuid.uuid4().hex[:12]
    workdir = SCRATCH_DIR / request_id
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        backend, payload, n_outputs = build_backend_payload(body, workdir)
        log(
            f"request {request_id}: task={payload['task']} variant={backend.variant} "
            f"target={payload['target']} conditions={len(payload['conditions'])} n={n_outputs}"
        )
        job = submit_and_wait(backend, payload, timeout_s=int(body.get("timeout_s", JOB_TIMEOUT_S)))
        submitted_at = time.time()

        out_prefix = (body.get("output") or {}).get("s3_uri_prefix") or OUTPUT_S3_PREFIX
        extract = body.get("extract") or {}
        outputs = []
        for i in range(n_outputs):
            data = fetch_variant_bytes(backend, job["id"], i)
            record: dict[str, Any] = {"variant": i, "bytes": len(data)}
            if out_prefix:
                bucket, key_prefix = parse_s3_uri(out_prefix.rstrip("/") + "/")
                key = f"{key_prefix}{request_id}/{payload['task']}-{i}.mp4"
                s3().put_object(Bucket=bucket, Key=key, Body=data, ContentType="video/mp4")
                record["s3_uri"] = f"s3://{bucket}/{key}"
                if extract:
                    record["artifacts"] = extract_artifacts(
                        data, workdir, extract, bucket, f"{key_prefix}{request_id}", i
                    )
            else:
                record["content_type"] = "video/mp4"
                record["base64"] = base64.b64encode(data).decode()
                if extract:
                    log("ignoring 'extract': it needs an output S3 prefix to write to")
            outputs.append(record)

        return {
            "status": "completed",
            "request_id": request_id,
            "task": payload["task"],
            "backend": {"variant": backend.variant, "port": backend.port, "gpus": backend.gpus},
            "job": {k: v for k, v in job.items() if k not in ("data", "content")},
            "outputs": outputs,
            "timings": {
                "total_s": round(time.time() - t0, 2),
                "generate_s": round(submitted_at - t0, 2),
                "download_s": round(time.time() - submitted_at, 2),
            },
            "request_echo": {
                "target": payload["target"],
                "num_outputs_per_prompt": n_outputs,
                "seed": payload.get("seed"),
                "num_inference_steps": payload.get("num_inference_steps"),
            },
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "minimax-h3-sagemaker-shim/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter access log
        if self.path != "/ping":
            log(f"http {self.address_string()} {fmt % args}")

    def _respond(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.split("?")[0] in ("/ping", "/health"):
            if _READY.is_set():
                self._respond(200, {"status": "ready", "variants": list(BACKENDS)})
            else:
                self._respond(
                    503,
                    {
                        "status": "failed" if _BOOT_ERROR else "loading",
                        "errors": _BOOT_ERROR,
                        "elapsed_s": round(time.time() - _START_TS, 1),
                    },
                )
            return
        if self.path.split("?")[0] == "/backends":
            self._respond(
                200,
                {
                    "ready": _READY.is_set(),
                    "errors": _BOOT_ERROR,
                    "backends": [
                        {
                            "variant": b.variant,
                            "port": b.port,
                            "gpus": b.gpus,
                            "alive": b.alive(),
                            "ready": b.ready,
                            "health_path": b.health_path,
                            "command": b.command(),
                        }
                        for b in BACKENDS.values()
                    ],
                },
            )
            return
        self._respond(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.split("?")[0] != "/invocations":
            self._respond(404, {"error": f"no route {self.path}"})
            return
        if not _READY.is_set():
            self._respond(503, {"error": "model is still loading", "errors": _BOOT_ERROR})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(body, dict):
                raise RequestError("request body must be a JSON object")
        except (ValueError, UnicodeDecodeError) as exc:
            self._respond(400, {"error": f"invalid JSON body: {exc}"})
            return

        try:
            self._respond(200, handle_invocation(body))
        except RequestError as exc:
            log(f"400 {exc}")
            self._respond(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - report, do not crash the worker
            log("500 during invocation:\n" + traceback.format_exc())
            self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})


def _shutdown(signum: int, _frame: Any) -> None:
    log(f"signal {signum}: stopping backends")
    for backend in BACKENDS.values():
        if backend.alive() and backend.proc is not None:
            backend.proc.terminate()
    sys.exit(0)


def main() -> None:
    log("MiniMax-H3 SageMaker shim starting")
    log(
        f"config: variants={VARIANTS} gpus_per_backend={GPUS_PER_BACKEND} "
        f"tp={TP_SIZE} ulysses={ULYSSES_DEGREE} mode={PERFORMANCE_MODE} "
        f"warmup_on_start={WARMUP_ON_START} output_prefix={OUTPUT_S3_PREFIX or '(inline base64)'}"
    )
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    threading.Thread(target=boot, daemon=True).start()

    socket.setdefaulttimeout(None)
    server = ThreadingHTTPServer(("0.0.0.0", BIND_PORT), Handler)
    server.daemon_threads = True
    log(f"listening on 0.0.0.0:{BIND_PORT} (/ping, /invocations, /backends)")
    server.serve_forever()


if __name__ == "__main__":
    main()
