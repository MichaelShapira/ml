#!/usr/bin/env python3
"""Stage FLUX 3 Image Klein + its prompt upsampler from the Hugging Face Hub into an S3 prefix, once.

The notebook runs this as a SageMaker Processing job (CPU instance, same Ray Serve DLC image as the
endpoint), so the ~31 GB transfer happens inside AWS instead of through the notebook's network and disk.
It also runs anywhere with `huggingface_hub` and `boto3`.

What it stages, read from the repo's own `modular_model_index.json` (not hard-coded):

* every pipeline component stored in the main repo (transformer/, vae/, text_encoder/, ...), at the
  subfolder the index names. The repo-root single-file package (model.safetensors, ae.safetensors) and
  the FP8/NVFP4 `variants/` are skipped: Diffusers loads the BF16 subfolders.
* every component the index points at ANOTHER repo (the prompt upsampler,
  black-forest-labs/flux-3-image-klein-pup), into a local subfolder named after the component.
  `_flux3_stage.json` records that mapping; the endpoint reads it to load those components offline.
* the repo's LICENSE.md / NOTICE.md / README.md, next to the weights.

Resumable: an object already in S3 with the same size is skipped. The manifest is written last, so its
presence means staging completed; a final S3 listing verifies every object size against it.

The Hugging Face token comes from $HF_TOKEN or, preferably, from a Secrets Manager secret
(`--hf-token-secret`), so it never appears in the Processing job's description.

Usage:
  python stage_weights.py --bucket B --prefix flux3-klein-g7e/model/weights --hf-token-secret NAME
  python stage_weights.py --bucket B --prefix P --dry-run          # list what would be staged
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

MAIN_REPO = "black-forest-labs/flux-3-image-klein"
INDEX = "modular_model_index.json"
MANIFEST = "_flux3_stage.json"
LICENSE_FILES = ("LICENSE.md", "NOTICE.md", "README.md")
SKIP_FILES = (".gitattributes",)


def log(message: str) -> None:
    print(f"[stage {dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {message}", flush=True)


def read_token(args) -> str | None:
    if args.hf_token_secret:
        import boto3

        value = boto3.client("secretsmanager", region_name=args.region).get_secret_value(
            SecretId=args.hf_token_secret
        )["SecretString"].strip()
        if value.startswith("{"):
            value = json.loads(value).get("HF_TOKEN", "")
        log(f"Hugging Face token read from Secrets Manager ({args.hf_token_secret})")
        return value or None
    token = os.environ.get("HF_TOKEN") or None
    log("Hugging Face token from $HF_TOKEN" if token else "no Hugging Face token (public repos only)")
    return token


def plan_staging(api, repo: str, token: str | None, scratch: Path) -> tuple[dict, list[dict], dict]:
    """(main repo info, groups to download, component overrides) derived from the repo's pipeline index."""
    from huggingface_hub import hf_hub_download

    info = api.model_info(repo, files_metadata=True, token=token)
    index_path = hf_hub_download(repo, INDEX, revision=info.sha, token=token, local_dir=scratch / "_index")
    index = json.loads(Path(index_path).read_text())
    log(f"{repo}@{info.sha[:12]}: pipeline {index.get('_class_name')} / {index.get('_blocks_class_name')}")

    sources: dict[tuple[str, str | None, str], list[str]] = {}
    for name, value in index.items():
        if not (isinstance(value, list) and len(value) == 3 and isinstance(value[2], dict)):
            continue
        spec = value[2]
        source = spec.get("pretrained_model_name_or_path") or repo
        key = (source, spec.get("revision"), spec.get("subfolder") or "")
        sources.setdefault(key, []).append(name)

    groups, overrides = [], {}
    main_files = {s.rfilename: s.size or 0 for s in info.siblings}
    for (source, revision, subfolder), names in sources.items():
        if source == repo:
            if not subfolder:
                raise SystemExit(f"components {names} live at the root of {repo}; this layout is not supported")
            files = {f: n for f, n in main_files.items() if f.startswith(subfolder + "/")}
            groups.append({"repo": repo, "revision": info.sha, "names": names, "files": files, "dest": None})
            continue
        ext = api.model_info(source, revision=revision, files_metadata=True, token=token)
        dest = "prompt_upsampler" if "prompt_upsampler" in names else names[0]
        prefix = subfolder + "/" if subfolder else ""
        files = {
            s.rfilename: s.size or 0
            for s in ext.siblings
            if s.rfilename.startswith(prefix) and Path(s.rfilename).name not in SKIP_FILES
        }
        groups.append({"repo": source, "revision": ext.sha, "names": names, "files": files, "dest": dest,
                       "strip": prefix})
        for name in names:
            overrides[name] = {"subfolder": dest, "source_repo": source, "source_revision": ext.sha,
                               "source_subfolder": subfolder}
    root_files = {f: main_files[f] for f in (INDEX, *LICENSE_FILES) if f in main_files}
    groups.append({"repo": repo, "revision": info.sha, "names": ["index", "license"], "files": root_files,
                   "dest": None})
    return {"repo": repo, "revision": info.sha, "index": index_path}, groups, overrides


def s3_key_for(group: dict, filename: str, prefix: str) -> str:
    if group["dest"] is None:
        relative = filename
    else:
        relative = f"{group['dest']}/{filename[len(group.get('strip', '')):]}"
    return f"{prefix.rstrip('/')}/{relative}"


def existing_sizes(s3, bucket: str, prefix: str) -> dict[str, int]:
    sizes = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            sizes[obj["Key"]] = obj["Size"]
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=MAIN_REPO)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True, help="S3 prefix for the weights, e.g. flux3-klein-g7e/model/weights")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    parser.add_argument("--hf-token-secret", default=os.environ.get("HF_TOKEN_SECRET") or None)
    parser.add_argument("--scratch", default=os.environ.get("STAGE_SCRATCH", "/tmp/flux3-stage"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import boto3
    from boto3.s3.transfer import TransferConfig
    from huggingface_hub import HfApi, snapshot_download

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    token = read_token(args)
    api = HfApi()
    s3 = boto3.client("s3", region_name=args.region)
    started = time.time()

    main_info, groups, overrides = plan_staging(api, args.repo, token, scratch)
    total = sum(sum(g["files"].values()) for g in groups)
    for g in groups:
        log(f"  {g['repo']}@{g['revision'][:12]} -> {g['dest'] or '(same paths)'}: {len(g['files'])} files, "
            f"{sum(g['files'].values()) / 1e9:.2f} GB  [{', '.join(g['names'])}]")
    log(f"total {total / 1e9:.2f} GB -> s3://{args.bucket}/{args.prefix.rstrip('/')}/")
    if args.dry_run:
        return 0

    already = existing_sizes(s3, args.bucket, args.prefix)
    transfer = TransferConfig(multipart_chunksize=64 * 1024**2, max_concurrency=16)
    staged: list[dict] = []
    for g in groups:
        todo = [f for f in g["files"] if already.get(s3_key_for(g, f, args.prefix)) != g["files"][f]]
        for f in g["files"]:
            key = s3_key_for(g, f, args.prefix)
            staged.append({"key": key, "path": key[len(args.prefix.rstrip("/")) + 1:], "size": g["files"][f],
                           "source": g["repo"]})
        if not todo:
            if g["files"]:
                log(f"{g['repo']}: all {len(g['files'])} files already in S3, skipping")
            continue
        local = scratch / g["repo"].replace("/", "--")
        t0 = time.time()
        snapshot_download(g["repo"], revision=g["revision"], allow_patterns=todo, local_dir=local, token=token,
                          max_workers=args.workers)
        size = sum(g["files"][f] for f in todo)
        log(f"{g['repo']}: downloaded {len(todo)} files, {size / 1e9:.2f} GB in {time.time() - t0:.0f}s")

        t0 = time.time()

        def upload(filename: str, group=g, root=local) -> str:
            key = s3_key_for(group, filename, args.prefix)
            s3.upload_file(str(root / filename), args.bucket, key, Config=transfer)
            return key

        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            for key in pool.map(upload, todo):
                log(f"  uploaded s3://{args.bucket}/{key}")
        log(f"{g['repo']}: uploaded {size / 1e9:.2f} GB in {time.time() - t0:.0f}s")

    manifest = {
        "repo": main_info["repo"],
        "revision": main_info["revision"],
        "staged_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "components": overrides,
        "files": staged,
        "total_bytes": total,
    }
    final = existing_sizes(s3, args.bucket, args.prefix)
    bad = [f["key"] for f in staged if final.get(f["key"]) != f["size"]]
    if bad:
        log(f"VERIFY FAILED: {len(bad)} objects missing or with the wrong size, e.g. {bad[:5]}")
        return 2
    key = f"{args.prefix.rstrip('/')}/{MANIFEST}"
    s3.put_object(Bucket=args.bucket, Key=key, Body=json.dumps(manifest, indent=1).encode(),
                  ContentType="application/json")
    log(f"verified {len(staged)} objects ({total / 1e9:.2f} GB); wrote s3://{args.bucket}/{key} "
        f"in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
