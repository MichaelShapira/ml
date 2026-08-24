#!/usr/bin/env python3
"""
Stage the MiniMax-H3 checkpoint from Hugging Face into S3, file by file.

Why not just `hf download --local-dir X && aws s3 sync X s3://...`?
Because the two SGLang partitions are 144.1 GB each (288 GB together) and the
machine that stages them is usually a modest notebook instance. This script
streams: download one file -> upload it -> delete it, with a small worker pool,
so peak local disk stays around `workers x largest_file` (~30-45 GB) instead of
288 GB.

It also writes `_h3_stage.json` next to the tree, recording the repo revision.
The container turns the staged tree into a warm Hugging Face cache entry for
that exact revision, which is what lets `sglang serve --model-path
MiniMaxAI/MiniMax-H3` resolve local weights offline. (The SGLang cookbook is
explicit that SGLang owns the checkpoint-directory mapping and you should not
point --model-path at a hand-downloaded subdirectory.)

Two transfer modes
------------------
``--stream`` (default) relays each file straight from the Hub into S3 through a
bounded in-memory buffer. **Nothing touches local disk.** This matters on a
SageMaker notebook instance, whose ML volume is a default gp3/gp2 EBS volume
capped around 125-250 MB/s: writing 288 GB down and reading it straight back up
makes the disk, not the network, the bottleneck. It also means the instance needs
no scratch space at all.

``--no-stream`` uses ``hf_hub_download`` to disk, then uploads, then deletes.
Slightly more robust to flaky connections (the Hub client retries and resumes
partial downloads) and it can use ``hf_transfer``'s parallel chunk downloads for
a single large file. Needs roughly ``workers x largest_file`` (~31 GB) of scratch
and a disk that can keep up.

Resumable either way: a file already present in S3 with a matching size is
skipped.

Usage
-----
  python stage_weights.py --bucket my-bucket --prefix minimax-h3/weights
  python stage_weights.py --bucket b --prefix p --variants fl2va        # 144 GB
  python stage_weights.py --bucket b --prefix p --no-stream --scratch /mnt/nvme
  python stage_weights.py --bucket b --prefix p --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ID = "MiniMaxAI/MiniMax-H3"

# The original (non-diffusers) checkpoint layout SGLang consumes. The repo also
# carries a diffusers-format conversion (transformer/, transformer_ref/,
# text_encoder/, vae/, audio_vae/) side by side; staging both would be ~500 GB
# for no benefit here.
VARIANT_DIRS = {"fl2va": "FL2VA", "ref2va": "Ref2VA"}
ROOT_FILES = ("model_index.json", "modular_model_index.json", "LICENSE", "README.md")


@dataclass
class Item:
    path: str
    size: int


def human(n: float) -> str:
    """Decimal units, to match how Hugging Face and S3 report sizes.

    (Binary units would print the 144.1 GB partition as 134.2 GiB, which reads
    like a missing file rather than a different base.)
    """
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def list_repo_items(variants: list[str], token: str | None) -> tuple[str, list[Item]]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.model_info(REPO_ID, files_metadata=False)
    revision = info.sha
    wanted_prefixes = tuple(VARIANT_DIRS[v] + "/" for v in variants)

    items: list[Item] = []
    for entry in api.list_repo_tree(REPO_ID, recursive=True, revision=revision, expand=True):
        path = getattr(entry, "path", None)
        if path is None or getattr(entry, "size", None) is None:
            continue  # directory
        if path in ROOT_FILES or path.startswith(wanted_prefixes):
            size = getattr(entry, "size", 0) or 0
            lfs = getattr(entry, "lfs", None)
            if lfs is not None and getattr(lfs, "size", None):
                size = lfs.size
            items.append(Item(path=path, size=size))
    items.sort(key=lambda it: -it.size)
    return revision, items


def existing_sizes(s3, bucket: str, prefix: str) -> dict[str, int]:
    out: dict[str, int] = {}
    token = None
    base = prefix.rstrip("/") + "/"
    while True:
        kwargs = {"Bucket": bucket, "Prefix": base}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            out[obj["Key"][len(base):]] = obj["Size"]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


# --------------------------------------------------------------------------- #
# streaming relay: Hub -> S3, no local disk
# --------------------------------------------------------------------------- #

# Deliberately stdlib-only HTTP.
#
# The first version of this used `huggingface_hub.utils.get_session()`. That
# helper returns a `requests.Session` on older huggingface_hub and an
# `httpx.Client` on 1.x, and httpx has no `stream=` keyword on `.get()`, so the
# relay died with `Client.get() got an unexpected keyword argument 'stream'` on
# every one of 166 files. Depending on an internal helper whose *return type*
# changed across a major version is exactly the trap the SKILL warns about, so
# this path now uses urllib and has no opinion about which HTTP library is
# installed.

_REDIRECT_CODES = (301, 302, 303, 307, 308)
_MAX_REDIRECTS = 6
_RETRIES = 3


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface redirects as HTTPError so we can handle them ourselves."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


class _ExactReader:
    """Turn a stream whose ``read(n)`` may return short reads into one that does not.

    ``http.client.HTTPResponse.read(n)`` is allowed to return fewer than ``n``
    bytes before EOF. s3transfer sizes multipart parts from what ``read`` hands
    back, so a short read could silently produce a truncated part. Looping here
    removes the question entirely.
    """

    def __init__(self, raw) -> None:
        self._raw = raw

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._raw.read()
        out = bytearray()
        while len(out) < size:
            chunk = self._raw.read(size - len(out))
            if not chunk:
                break
            out += chunk
        return bytes(out)


def _hub_url(path: str, revision: str) -> str:
    try:
        from huggingface_hub import hf_hub_url

        return hf_hub_url(REPO_ID, path, revision=revision)
    except Exception:  # noqa: BLE001 - the URL format is public and stable
        return f"https://huggingface.co/{REPO_ID}/resolve/{revision}/{urllib.parse.quote(path)}"


def _open_hub_stream(url: str, token: str | None, read_timeout: int = 180):
    """GET a Hub file, following redirects by hand.

    Redirects are handled manually for one reason: the Hub 302s large files to a
    CDN with a pre-signed URL, and forwarding the ``Authorization`` header to
    that host makes it reject the request. So the token is only sent to the
    original host.
    """
    origin = urllib.parse.urlparse(url).netloc
    for _ in range(_MAX_REDIRECTS):
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", "minimax-h3-stage/1.0")
        # No gzip: HTTPResponse does not transparently decompress, and safetensors
        # are incompressible anyway.
        req.add_header("Accept-Encoding", "identity")
        if token and urllib.parse.urlparse(url).netloc == origin:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            return _OPENER.open(req, timeout=read_timeout)
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location") if exc.headers else None
            if exc.code in _REDIRECT_CODES and location:
                url = urllib.parse.urljoin(url, location)
                exc.close()
                continue
            raise
    raise RuntimeError(f"more than {_MAX_REDIRECTS} redirects for {url}")


def _relay_to_s3(s3, transfer, item: Item, revision: str, bucket: str, key: str, token: str | None) -> None:
    """Stream one file Hub -> S3 without touching local disk.

    ``upload_fileobj`` reads a non-seekable stream in ``multipart_chunksize``
    slices, so memory stays bounded at roughly ``chunksize x max_concurrency``
    per file however large the file is.

    A retry restarts the whole file: boto3 aborts the failed multipart upload, so
    there is no partial object left behind to resume from.
    """
    url = _hub_url(item.path, revision)
    last: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            with _open_hub_stream(url, token) as resp:
                length = resp.headers.get("Content-Length")
                if length is not None and int(length) != item.size:
                    raise OSError(
                        f"size mismatch before upload: Hub says {length}, manifest says {item.size}"
                    )
                s3.upload_fileobj(_ExactReader(resp), bucket, key, Config=transfer)
            return
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            last = exc
            if attempt < _RETRIES:
                time.sleep(2 ** attempt)
    raise last if last else RuntimeError("relay failed without an exception")


def stage(
    bucket: str,
    prefix: str,
    variants: list[str],
    scratch: Path,
    workers: int,
    token: str | None,
    dry_run: bool,
    stream: bool = True,
) -> str:
    import boto3
    from boto3.s3.transfer import TransferConfig

    s3 = boto3.client("s3")
    if stream:
        # Smaller parts + lower concurrency: the source is one HTTP connection,
        # so a huge in-flight window buys nothing and just costs memory.
        transfer = TransferConfig(
            multipart_threshold=32 * 1024**2, multipart_chunksize=32 * 1024**2, max_concurrency=4
        )
    else:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        transfer = TransferConfig(
            multipart_threshold=64 * 1024**2, multipart_chunksize=64 * 1024**2, max_concurrency=8
        )
    base = prefix.rstrip("/") + "/"

    revision, items = list_repo_items(variants, token)
    total = sum(it.size for it in items)
    print(f"repo      : {REPO_ID}@{revision[:12]}")
    print(f"variants  : {variants}")
    print(f"files     : {len(items)}  total {human(total)}")
    print(f"target    : s3://{bucket}/{base}")
    if stream:
        print(f"mode      : stream (no local disk), {workers} workers, "
              f"~{human(workers * 4 * 32 * 1024**2)} peak RAM")
    else:
        print(f"mode      : via disk, scratch {scratch}, "
              f"peak ~{human(workers * items[0].size)} with {workers} workers")

    have = existing_sizes(s3, bucket, base)
    todo = [it for it in items if have.get(it.path) != it.size]
    skipped = len(items) - len(todo)
    remaining = sum(it.size for it in todo)
    print(f"already staged: {skipped} files; to transfer: {len(todo)} files / {human(remaining)}")

    if dry_run:
        for it in todo[:15]:
            print(f"  would stage {human(it.size):>12}  {it.path}")
        if len(todo) > 15:
            print(f"  ... and {len(todo) - 15} more")
        return revision

    if not stream:
        scratch.mkdir(parents=True, exist_ok=True)
    work: queue.Queue[Item | None] = queue.Queue()
    for it in todo:
        work.put(it)
    for _ in range(workers):
        work.put(None)

    done_bytes = 0
    done_files = 0
    lock = threading.Lock()
    errors: list[str] = []
    t0 = time.time()

    def worker(wid: int) -> None:
        nonlocal done_bytes, done_files
        while True:
            item = work.get()
            if item is None:
                return
            local = None
            try:
                if stream:
                    _relay_to_s3(s3, transfer, item, revision, bucket, base + item.path, token)
                else:
                    from huggingface_hub import hf_hub_download

                    local = Path(
                        hf_hub_download(
                            REPO_ID,
                            filename=item.path,
                            revision=revision,
                            local_dir=str(scratch / f"w{wid}"),
                            token=token,
                        )
                    )
                    s3.upload_file(str(local), bucket, base + item.path, Config=transfer)
                ok = True
            except Exception as exc:  # noqa: BLE001 - collect and report all
                ok = False
                with lock:
                    errors.append(f"{item.path}: {type(exc).__name__}: {exc}")
            finally:
                if local is not None:
                    try:
                        local.unlink(missing_ok=True)
                    except OSError:
                        pass
                with lock:
                    done_files += 1
                    # Only successful bytes count. Counting failures here is how
                    # the first version of this script reported "10.4 TB/s" and
                    # "eta 0.0 min" while transferring precisely nothing.
                    if ok:
                        done_bytes += item.size
                    elapsed = max(time.time() - t0, 1e-3)
                    rate = done_bytes / elapsed
                    eta = (remaining - done_bytes) / rate if rate else 0
                    flag = "" if ok else "  FAILED"
                    print(
                        f"  [{done_files}/{len(todo)}] {human(done_bytes)}/{human(remaining)} "
                        f"@ {human(rate)}/s  eta {eta / 60:5.1f} min  {item.path}{flag}",
                        flush=True,
                    )

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not stream:
        shutil.rmtree(scratch, ignore_errors=True)

    if errors:
        shown = errors[:20]
        print(f"\n{len(errors)} FAILURES (showing {len(shown)}):", file=sys.stderr)
        for e in shown:
            print("  " + e, file=sys.stderr)
        print("\nNothing was marked complete. Re-run to retry only what is missing.", file=sys.stderr)
        raise SystemExit(1)

    # Verify against S3 rather than trusting the transfer loop. One LIST call, and
    # it catches a truncated object that the loop thought succeeded - the failure
    # mode that matters most for a streamed upload.
    print("\nverifying against S3...")
    actual = existing_sizes(s3, bucket, base)
    bad = [
        (it.path, it.size, actual.get(it.path))
        for it in items
        if actual.get(it.path) != it.size
    ]
    if bad:
        print(f"{len(bad)} object(s) missing or wrong size:", file=sys.stderr)
        for path, want, got in bad[:20]:
            print(f"  {path}: expected {human(want)}, got {'missing' if got is None else human(got)}",
                  file=sys.stderr)
        print("\nRe-run to fix; matching files are skipped.", file=sys.stderr)
        raise SystemExit(1)
    print(f"verified {len(items)} objects, {human(sum(it.size for it in items))}")

    marker = {
        "repo_id": REPO_ID,
        "revision": revision,
        "variants": variants,
        "files": len(items),
        "total_bytes": total,
        "mode": "stream" if stream else "disk",
        "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    s3.put_object(
        Bucket=bucket,
        Key=base + "_h3_stage.json",
        Body=json.dumps(marker, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> s3://{bucket}/{base}")
    print(f"revision recorded in _h3_stage.json: {revision}")
    return revision


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="minimax-h3/weights")
    ap.add_argument(
        "--variants",
        default="fl2va,ref2va",
        help="comma separated: fl2va (t2va+fl2va, 144 GB), ref2va (144 GB)",
    )
    ap.add_argument("--scratch", default="/tmp/h3-stage", help="only used with --no-stream")
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="parallel transfers (default: 6 streaming, 3 via disk)",
    )
    ap.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="download to disk first instead of relaying Hub -> S3 in memory",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.set_defaults(stream=True)
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANT_DIRS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {sorted(VARIANT_DIRS)}")

    stage(
        bucket=args.bucket,
        prefix=args.prefix,
        variants=variants,
        scratch=Path(args.scratch),
        workers=args.workers or (6 if args.stream else 3),
        token=os.environ.get("HF_TOKEN"),
        dry_run=args.dry_run,
        stream=args.stream,
    )


if __name__ == "__main__":
    main()
