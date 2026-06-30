#!/usr/bin/env python3
"""Progress / log monitor for the SAM 3D Objects SageMaker workflow.

Run this in a SEPARATE terminal while the notebook builds the image or deploys
the endpoint. It is a one-shot snapshot by default; add --follow to poll, which
is BOUNDED by --timeout and stops automatically at any terminal state (no
endless loop).

Subcommands
-----------
  build      Latest CodeBuild image build (status, phases, CloudWatch logs).
  endpoint   Endpoint status + FailureReason + /aws/sagemaker/Endpoints logs.
  failures   Dump recent async failure objects from the failure S3 prefix.

Examples
--------
  python monitor.py build --follow
  python monitor.py endpoint --name sam3d-objects-g6e --follow
  python monitor.py endpoint            # single snapshot + recent logs
  python monitor.py failures
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import boto3

ENDPOINT_DEFAULT = "sam3d-objects-g6e"
ECR_REPO_DEFAULT = "sam3d-objects"
BUILD_PROJECT_PREFIX = "sagemaker-studio"   # sm-docker names its CodeBuild project this
FAILURE_PREFIX = "sam3d-failures/"
ENDPOINT_TERMINAL = {"InService", "Failed", "OutOfService", "DeleteFailed"}
BUILD_TERMINAL = {"SUCCEEDED", "FAILED", "FAULT", "TIMED_OUT", "STOPPED"}


def _region():
    s = boto3.Session()
    if not s.region_name:
        sys.exit("No AWS region configured. Set AWS_DEFAULT_REGION or run `aws configure`.")
    return s.region_name


def _ts(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H:%M:%S")


def _tail_group(logs, group, start_ms, seen, stream=None):
    """Print new events from a log group since start_ms; return (new_start_ms, seen)."""
    kwargs = {"logGroupName": group, "startTime": start_ms, "limit": 10000}
    if stream:
        kwargs["logStreamNames"] = [stream]
    newest = start_ms
    try:
        paginator = logs.get_paginator("filter_log_events")
        for page in paginator.paginate(**kwargs):
            for e in page.get("events", []):
                eid = e.get("eventId")
                if eid in seen:
                    continue
                seen.add(eid)
                print(f"  [{_ts(e['timestamp'])}] {e['message'].rstrip()}")
                newest = max(newest, e["timestamp"] + 1)
    except logs.exceptions.ResourceNotFoundException:
        print(f"  (log group {group} not created yet)")
    except Exception as exc:  # logs perms / transient
        print(f"  (could not read logs: {exc})")
    return newest, seen


# --------------------------------------------------------------------------- build

def _get_build(cb, build_id):
    """Fetch a single build dict, or None if it's missing/expired."""
    try:
        builds = cb.batch_get_builds(ids=[build_id]).get("builds", [])
    except Exception as exc:
        print(f"  (batch_get_builds failed: {exc})")
        return None
    return builds[0] if builds else None


def _latest_build_id(cb, project_prefix):
    """Return (project_name, build_id) for the most recently started build across
    projects whose name starts with project_prefix, or (None, None)."""
    try:
        projects = []
        for page in cb.get_paginator("list_projects").paginate():
            projects.extend(page.get("projects", []))
    except Exception as exc:
        print(f"Could not list CodeBuild projects: {exc}")
        return None, None
    cands = [p for p in projects if p.startswith(project_prefix)]
    if not cands:
        return None, None
    best_proj, best_id, best_start = None, None, -1.0
    for proj in cands:
        try:
            ids = cb.list_builds_for_project(projectName=proj, sortOrder="DESCENDING").get("ids", [])
        except Exception:
            continue
        if not ids:
            continue
        b = _get_build(cb, ids[0])
        if not b:
            continue
        start = b.get("startTime")
        start = start.timestamp() if start else 0.0
        if start > best_start:
            best_proj, best_id, best_start = proj, ids[0], start
    return best_proj, best_id


def cmd_build(args):
    region = _region()
    cb = boto3.client("codebuild", region_name=region)
    logs = boto3.client("logs", region_name=region)

    proj, build_id = _latest_build_id(cb, args.project)
    if not build_id:
        print(f"No started build found for a CodeBuild project beginning with "
              f"'{args.project}'.\nHas section 5 (the sm-docker build) been launched yet? "
              f"It can take ~30s for the project + build to appear.")
        return

    start_ms = 0
    seen = set()
    deadline = time.time() + args.timeout
    while True:
        build = _get_build(cb, build_id)
        if not build:
            print(f"Build {build_id} not found (it may have aged out of CodeBuild history).")
            return
        status = build.get("status", "UNKNOWN")
        phase = build.get("currentPhase", "?")
        print(f"\n=== CodeBuild {proj} | build {build_id.split(':')[-1][:8]} "
              f"| status={status} phase={phase} ===")
        log_info = build.get("logs", {}) or {}
        group, stream = log_info.get("groupName"), log_info.get("streamName")
        if group and stream:
            start_ms, seen = _tail_group(logs, group, start_ms, seen, stream=stream)
        else:
            print("  (build logs not available yet — provisioning the build environment)")

        if status in BUILD_TERMINAL:
            print(f"\nBuild finished: {status}")
            if status != "SUCCEEDED":
                print("Tip: read the failed phase above. If it TIMED_OUT during the "
                      "flash_attn/pytorch3d compile, raise the CodeBuild project timeout and re-run.")
            return
        if not args.follow or time.time() > deadline:
            if args.follow:
                print(f"\n(stopped after --timeout {args.timeout}s; build still {status})")
            return
        time.sleep(args.interval)


# ----------------------------------------------------------------------- endpoint

def cmd_endpoint(args):
    region = _region()
    sm = boto3.client("sagemaker", region_name=region)
    logs = boto3.client("logs", region_name=region)
    group = f"/aws/sagemaker/Endpoints/{args.name}"

    # Start logs from the last `since` minutes on the first pass.
    start_ms = int((time.time() - args.since * 60) * 1000)
    seen = set()
    deadline = time.time() + args.timeout
    last_status = None
    while True:
        try:
            d = sm.describe_endpoint(EndpointName=args.name)
        except sm.exceptions.ClientError as exc:
            print(f"describe-endpoint failed: {exc}")
            print("The endpoint may not be created yet (section 6).")
            return
        status = d["EndpointStatus"]
        reason = d.get("FailureReason", "")
        if status != last_status:
            print(f"\n=== Endpoint {args.name} | status={status} "
                  f"| {datetime.now().strftime('%H:%M:%S')} ===")
            if reason:
                print("  FailureReason:", reason)
            last_status = status
        start_ms, seen = _tail_group(logs, group, start_ms, seen)

        if status in ENDPOINT_TERMINAL:
            print(f"\nEndpoint reached terminal state: {status}")
            if status == "Failed":
                print("Read FailureReason + logs above. Common causes: image pull denied "
                      "(IAM section 2), OOM (<32GB VRAM), or a model_fn import error.")
            elif status == "InService":
                print("Ready to invoke (notebook section 9).")
            return
        if not args.follow or time.time() > deadline:
            if args.follow:
                print(f"\n(stopped after --timeout {args.timeout}s; endpoint still {status})")
            return
        time.sleep(args.interval)


# ----------------------------------------------------------------------- failures

def cmd_failures(args):
    region = _region()
    s3 = boto3.client("s3", region_name=region)
    sts = boto3.client("sts", region_name=region)
    bucket = args.bucket or f"sagemaker-{region}-{sts.get_caller_identity()['Account']}"
    print(f"Scanning s3://{bucket}/{FAILURE_PREFIX} ...")
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=FAILURE_PREFIX)
    except Exception as exc:
        print("Could not list failures:", exc)
        return
    objs = sorted(resp.get("Contents", []), key=lambda o: o["LastModified"], reverse=True)
    if not objs:
        print("No async failures recorded. (Good — or none have run yet.)")
        return
    for o in objs[: args.limit]:
        print(f"\n--- {o['Key']}  ({o['LastModified']:%Y-%m-%d %H:%M:%S}) ---")
        body = s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read().decode("utf-8", "ignore")
        print(body[:4000])


def main():
    p = argparse.ArgumentParser(description="Monitor SAM 3D Objects build / deploy progress.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Latest CodeBuild image build status + logs.")
    pb.add_argument("--project", default=BUILD_PROJECT_PREFIX, help="CodeBuild project name prefix.")
    pb.add_argument("--follow", action="store_true", help="Poll until done or --timeout.")
    pb.add_argument("--interval", type=int, default=15)
    pb.add_argument("--timeout", type=int, default=5400, help="Max seconds to follow.")
    pb.set_defaults(func=cmd_build)

    pe = sub.add_parser("endpoint", help="Endpoint status + CloudWatch logs.")
    pe.add_argument("--name", default=ENDPOINT_DEFAULT)
    pe.add_argument("--follow", action="store_true", help="Poll until terminal or --timeout.")
    pe.add_argument("--interval", type=int, default=20)
    pe.add_argument("--timeout", type=int, default=3600, help="Max seconds to follow.")
    pe.add_argument("--since", type=int, default=20, help="Show logs from the last N minutes.")
    pe.set_defaults(func=cmd_endpoint)

    pf = sub.add_parser("failures", help="Dump recent async failure objects.")
    pf.add_argument("--bucket", default=None, help="Override (default: sagemaker-<region>-<account>).")
    pf.add_argument("--limit", type=int, default=5)
    pf.set_defaults(func=cmd_failures)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
