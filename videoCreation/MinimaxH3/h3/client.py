"""
Client for the MiniMax-H3 SageMaker *asynchronous* endpoint.

Why async and not real-time: a 5-second 768p clip at 50 denoising steps takes
minutes on 4xH100, and a SageMaker real-time endpoint caps a single response at
60 seconds. Async inference reads the request from S3, lets the container work
for up to ``InvocationTimeoutSeconds`` (max 3600), and writes the response to S3.

So one ``generate()`` call is: write request JSON to S3 -> invoke_endpoint_async
-> poll for the output (or failure) object -> download the MP4s.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "keyframe",
    "image_ref",
    "video_ref",
    "audio_ref",
    "Result",
    "H3AsyncEndpoint",
]


# --------------------------------------------------------------------------- #
# condition builders
# --------------------------------------------------------------------------- #


def _uri_field(source: str | Path) -> dict[str, Any]:
    text = str(source)
    if text.startswith(("s3://", "http://", "https://", "file://", "data:")):
        return {"uri": text}
    # A local path only means something inside the container, so inline it.
    from .media import data_uri

    return {"uri": data_uri(text)}


def keyframe(source: str | Path, *, frame_index: int | None = None) -> dict[str, Any]:
    """A first-frame / last-frame conditioning image for the ``fl2va`` task.

    ``frame_index=0`` marks the opening frame. Omit it for the closing frame and
    the server places it at the end of the clip.
    """
    cond: dict[str, Any] = {"type": "image", "role": "keyframe", **_uri_field(source)}
    if frame_index is not None:
        cond["frame_index"] = frame_index
    return cond


def image_ref(source: str | Path) -> dict[str, Any]:
    """An omni-reference image (``<Picture N>`` / a subject's appearance)."""
    return {"type": "image", "role": "reference", **_uri_field(source)}


def video_ref(source: str | Path, *, start_time_seconds: float | None = None) -> dict[str, Any]:
    """An omni-reference video clip (``<Video N>``): editing source, motion, rhythm.

    ``start_time_seconds`` seeks into the clip; its visual and audio streams are
    always sought together.
    """
    cond: dict[str, Any] = {"type": "video", "role": "reference", **_uri_field(source)}
    if start_time_seconds is not None:
        cond["start_time_seconds"] = start_time_seconds
    return cond


def audio_ref(source: str | Path) -> dict[str, Any]:
    """An omni-reference audio clip (``<Audio N>``): voice timbre, music, beat.

    Audio cannot be the only reference - H3 requires it alongside an image or
    video reference.
    """
    return {"type": "audio", "role": "reference", **_uri_field(source)}


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #


@dataclass
class Result:
    """A finished generation. The clips stay in S3 unless you ask for them.

    Nothing is downloaded by default: ``show()`` renders from a presigned URL, so
    the browser streams straight from S3 and no MP4 accumulates on the machine
    running the notebook. Sixteen demos would otherwise be a few hundred MB of
    local files plus the same again embedded as base64 in the .ipynb.
    """

    request_id: str
    task: str
    metadata: dict[str, Any]
    s3_uris: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0
    output_location: str = ""
    _s3: Any = None

    # -------------------------------------------------------------- accessors

    @property
    def s3_uri(self) -> str:
        if not self.s3_uris:
            raise RuntimeError("no video was produced")
        return self.s3_uris[0]

    @property
    def generate_s(self) -> float:
        return float(self.metadata.get("timings", {}).get("generate_s", 0.0))

    def summary(self) -> str:
        echo = self.metadata.get("request_echo", {})
        target = echo.get("target", {})
        return (
            f"{self.task}: {len(self.s3_uris)} clip(s), "
            f"{target.get('duration_seconds', '?')}s @ short edge {target.get('short_edge', '?')} "
            f"({target.get('aspect_ratio', '?')}), seed={echo.get('seed')}, "
            f"steps={echo.get('num_inference_steps')}, "
            f"generate {self.generate_s:.0f}s / wall {self.elapsed_s:.0f}s"
        )

    def as_state(self) -> dict[str, Any]:
        """A small JSON-safe record for :class:`h3.state.State`.

        Includes ``request_echo`` so a rehydrated Result can still describe itself
        accurately - without it, ``summary()`` after a kernel restart reports
        "?s @ short edge ?".
        """
        return {
            "request_id": self.request_id,
            "task": self.task,
            "s3_uris": list(self.s3_uris),
            "artifacts": dict(self.artifacts),
            "output_location": self.output_location,
            "generate_s": self.generate_s,
            "request_echo": self.metadata.get("request_echo", {}),
            "summary": self.summary(),
        }

    # ---------------------------------------------------------------- display

    def presign(self, uri: str | None = None, expires_s: int = 6 * 3600) -> str:
        if self._s3 is None:
            raise RuntimeError("this Result has no S3 client attached")
        target = uri or self.s3_uri
        parsed = urllib.parse.urlparse(target)
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": parsed.netloc, "Key": parsed.path.lstrip("/")},
            ExpiresIn=expires_s,
        )

    def show(self, width: int = 640, expires_s: int = 6 * 3600) -> None:
        """Play the clip(s) inline, streamed from S3. Downloads nothing."""
        from IPython.display import HTML, display

        for uri in self.s3_uris:
            url = self.presign(uri, expires_s)
            name = uri.rsplit("/", 1)[-1]
            display(HTML(
                f"<p style='font:600 13px sans-serif;margin:6px 0 2px'>{name} "
                f"<span style='font-weight:400;color:#666'>(streamed from S3)</span></p>"
                f'<video width="{width}" controls playsinline preload="metadata" src="{url}"></video>'
            ))

    def download(self, dest_dir: str | Path = ".") -> list[Path]:
        """Explicitly pull the clips local. Only call this when you want the bytes."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        out = []
        for i, uri in enumerate(self.s3_uris):
            parsed = urllib.parse.urlparse(uri)
            path = dest / f"{self.request_id}-{i}.mp4"
            self._s3.download_file(parsed.netloc, parsed.path.lstrip("/"), str(path))
            out.append(path)
        return out


# --------------------------------------------------------------------------- #
# endpoint
# --------------------------------------------------------------------------- #


class H3AsyncEndpoint:
    """Thin, explicit client for the async endpoint.

    Parameters
    ----------
    endpoint_name:
        SageMaker endpoint name.
    io_s3_prefix:
        ``s3://bucket/prefix`` used for request payloads, endpoint responses and
        (by default) the generated MP4s.
    download_dir:
        Local directory that finished clips are written to.
    invocation_timeout_s:
        Passed through as ``InvocationTimeoutSeconds``. SageMaker's ceiling is
        3600; a 15-second clip at 50 steps needs most of a 10-minute budget on
        4xH100, so leave this generous.
    """

    def __init__(
        self,
        endpoint_name: str,
        io_s3_prefix: str,
        *,
        region_name: str | None = None,
        invocation_timeout_s: int = 3600,
        boto_session: Any | None = None,
    ) -> None:
        import boto3

        session = boto_session or boto3.Session(region_name=region_name)
        self.endpoint_name = endpoint_name
        self.region = session.region_name
        self.sm = session.client("sagemaker")
        self.smr = session.client("sagemaker-runtime")
        self.s3 = session.client("s3")
        self.logs = session.client("logs")

        if not io_s3_prefix.startswith("s3://"):
            raise ValueError(f"io_s3_prefix must start with s3://, got {io_s3_prefix!r}")
        self.io_s3_prefix = io_s3_prefix.rstrip("/")
        self.bucket, _, self.key_prefix = self.io_s3_prefix[5:].partition("/")
        self.key_prefix = self.key_prefix.rstrip("/")
        self.invocation_timeout_s = invocation_timeout_s

    # ---------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        desc = self.sm.describe_endpoint(EndpointName=self.endpoint_name)
        return {
            "status": desc["EndpointStatus"],
            "reason": desc.get("FailureReason"),
            "config": desc.get("EndpointConfigName"),
            "instances": [
                {
                    "variant": v["VariantName"],
                    "instance_type": v.get("CurrentInstanceCount"),
                    "current": v.get("CurrentInstanceCount"),
                    "desired": v.get("DesiredInstanceCount"),
                }
                for v in desc.get("ProductionVariants", [])
            ],
        }

    def tail_logs(self, minutes: int = 15, limit: int = 60, filter_pattern: str = "") -> list[str]:
        """Return recent endpoint log lines.

        A *missing* log group means the container never booted - the failure was
        in SageMaker's managed bootstrap (image pull, model download, capacity),
        which is a different problem from a crashing container.
        """
        group = f"/aws/sagemaker/Endpoints/{self.endpoint_name}"
        try:
            kwargs: dict[str, Any] = {
                "logGroupName": group,
                "startTime": int((time.time() - minutes * 60) * 1000),
                "limit": limit,
            }
            if filter_pattern:
                kwargs["filterPattern"] = filter_pattern
            events = self.logs.filter_log_events(**kwargs).get("events", [])
        except self.logs.exceptions.ResourceNotFoundException:
            return [f"(no log group {group} yet - the container has not started)"]
        return [e["message"].rstrip() for e in events]

    # ------------------------------------------------------------- invoking

    def _put_request(self, payload: dict[str, Any], request_id: str) -> str:
        key = f"{self.key_prefix}/requests/{request_id}.json" if self.key_prefix else f"requests/{request_id}.json"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(payload).encode(),
            ContentType="application/json",
        )
        return f"s3://{self.bucket}/{key}"

    def _head(self, uri: str) -> bool:
        parsed = urllib.parse.urlparse(uri)
        try:
            self.s3.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
            return True
        except self.s3.exceptions.ClientError:
            return False

    def _get_json(self, uri: str) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(uri)
        body = self.s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
        return json.loads(body.decode())

    def build_payload(
        self,
        *,
        prompt: Any,
        task: str | None = None,
        conditions: Sequence[dict[str, Any]] | None = None,
        duration_s: float = 5.0,
        aspect_ratio: str | None = None,
        short_edge: int = 768,
        seed: int | None = 0,
        num_inference_steps: int | None = 50,
        flow_shift: float | None = 12.0,
        audio_flow_shift: float | None = 3.0,
        num_outputs: int = 1,
        extract: bool | dict[str, Any] = False,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the request the endpoint expects.

        ``prompt`` accepts a plain string or a ``BasePrompt`` / ``RefPrompt``
        from :mod:`h3.prompts`; those also supply the ``task`` and duration, so
        the two cannot silently disagree.

        ``extract=True`` asks the endpoint to also write the clip's first frame,
        last frame and soundtrack to S3. That is what lets the keyframe and
        reference demos chain off a generated clip without anything ever being
        downloaded: the container already has ffmpeg, so it does the work next to
        the bytes instead of shipping them to the client and back.
        """
        wire_task = task
        if hasattr(prompt, "render"):
            wire_task = wire_task or getattr(prompt, "wire_task", None)
            duration_s = getattr(prompt, "duration_s", duration_s)
            prompt_text = prompt.render()
        else:
            prompt_text = str(prompt)
        wire_task = wire_task or "t2va"

        if aspect_ratio is None:
            aspect_ratio = "16:9" if wire_task == "t2va" else "auto"

        payload: dict[str, Any] = {
            "task": wire_task,
            "prompt": prompt_text,
            "conditions": list(conditions or []),
            "target": {
                "short_edge": short_edge,
                "aspect_ratio": aspect_ratio,
                "duration_seconds": float(duration_s),
            },
            "num_outputs_per_prompt": num_outputs,
            "output": {"s3_uri_prefix": f"{self.io_s3_prefix}/generated"},
        }
        for key, value in (
            ("seed", seed),
            ("num_inference_steps", num_inference_steps),
            ("flow_shift", flow_shift),
            ("audio_flow_shift", audio_flow_shift),
        ):
            if value is not None:
                payload[key] = value
        if extract:
            payload["extract"] = (
                {"first_frame": True, "last_frame": True, "audio": True}
                if extract is True
                else extract
            )
        if extra_body:
            payload["extra_body"] = extra_body
        return payload

    def generate(
        self,
        *,
        prompt: Any,
        name: str | None = None,
        poll_interval_s: float = 5.0,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Result:
        """Submit one generation and block until the MP4s are on disk."""
        payload = self.build_payload(prompt=prompt, **kwargs)
        request_id = f"{name or payload['task']}-{uuid.uuid4().hex[:8]}"
        payload["request_id"] = request_id
        input_location = self._put_request(payload, request_id)

        t0 = time.time()
        response = self.smr.invoke_endpoint_async(
            EndpointName=self.endpoint_name,
            InputLocation=input_location,
            ContentType="application/json",
            Accept="application/json",
            InvocationTimeoutSeconds=self.invocation_timeout_s,
            InferenceId=request_id,
        )
        output_location = response["OutputLocation"]
        failure_location = response.get("FailureLocation", "")
        if verbose:
            print(f"[{request_id}] queued -> {output_location}")

        deadline = t0 + self.invocation_timeout_s + 300
        last_note = 0.0
        while time.time() < deadline:
            if self._head(output_location):
                break
            if failure_location and self._head(failure_location):
                detail = self._get_json(failure_location)
                raise RuntimeError(f"[{request_id}] endpoint reported failure: {json.dumps(detail)[:2000]}")
            if verbose and time.time() - last_note > 60:
                last_note = time.time()
                print(f"[{request_id}] still running, {time.time() - t0:.0f}s elapsed")
            time.sleep(poll_interval_s)
        else:
            raise TimeoutError(f"[{request_id}] no output after {time.time() - t0:.0f}s")

        body = self._get_json(output_location)
        if body.get("error"):
            raise RuntimeError(f"[{request_id}] {body['error']}")

        result = Result(
            request_id=request_id,
            task=body.get("task", payload["task"]),
            metadata=body,
            elapsed_s=time.time() - t0,
            output_location=output_location,
            _s3=self.s3,
        )
        for record in body.get("outputs", []):
            if record.get("s3_uri"):
                result.s3_uris.append(record["s3_uri"])
            elif record.get("base64"):
                # Only happens if the endpoint has no H3_OUTPUT_S3_PREFIX. Park the
                # bytes in S3 ourselves so the rest of the notebook has one contract.
                import base64 as _b64

                key = f"{self.key_prefix}/generated/{request_id}/{result.task}-{record.get('variant', 0)}.mp4"
                self.s3.put_object(
                    Bucket=self.bucket, Key=key,
                    Body=_b64.b64decode(record["base64"]), ContentType="video/mp4",
                )
                result.s3_uris.append(f"s3://{self.bucket}/{key}")
            result.artifacts.update(record.get("artifacts") or {})

        if verbose:
            print(f"[{request_id}] {result.summary()}")
            if result.artifacts:
                print(f"[{request_id}] artifacts: {', '.join(sorted(result.artifacts))}")
        return result

    def rehydrate(self, record: dict[str, Any]) -> Result:
        """Rebuild a :class:`Result` from a :class:`h3.state.State` entry.

        This is what lets you jump into the middle of the notebook: the clip is
        still in S3, so there is no reason to regenerate it just to look at it.
        """
        return Result(
            request_id=record.get("request_id", "?"),
            task=record.get("task", "?"),
            metadata={
                "timings": {"generate_s": record.get("generate_s", 0.0)},
                "request_echo": record.get("request_echo", {}),
            },
            s3_uris=list(record.get("s3_uris") or []),
            artifacts=dict(record.get("artifacts") or {}),
            output_location=record.get("output_location", ""),
            _s3=self.s3,
        )

    def generate_batch(self, jobs: Iterable[dict[str, Any]], *, max_workers: int = 4) -> list[Result]:
        """Submit several generations at once.

        The endpoint's own queue serialises them onto the single instance, so
        this shortens *your* wait, not the GPU time. Useful for a sweep.
        """
        jobs = list(jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(self.generate, verbose=False, **job) for job in jobs]
            results = []
            for future in futures:
                results.append(future.result())
        return results
