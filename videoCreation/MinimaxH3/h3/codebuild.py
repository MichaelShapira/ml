"""
Build the serving image in AWS CodeBuild, for hosts with no Docker daemon.

Why this exists: SageMaker **Studio** has no Docker daemon unless the domain sets
`EnableDockerAccess`, and even then a Studio space's EBS (100 GB by default) is a
poor place to extract a ~30 GB image. A SageMaker **notebook instance** has Docker
but puts its data-root on the fixed ~100 GB root volume, which usually has 10-20 GB
free. So on both of the obvious places to run the notebook, a local `docker build`
of `lmsysorg/sglang:dev` fails.

CodeBuild sidesteps all of it: 128 GB of build disk, Docker preinstalled, and the
image is pushed straight to ECR.

This is deliberately not `sagemaker-studio-image-build`. That package wraps the
same idea but fixes the compute type at the default (64 GB of disk, which is too
tight here) and adds a dependency for ~150 lines of boto3.

Everything is idempotent: re-running updates the project rather than failing.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

__all__ = ["ensure_codebuild_role", "build_image", "BUILDSPEC"]

# CodeBuild compute types and their disk, which is the binding constraint:
#   SMALL 64 GB | MEDIUM 128 GB | LARGE 128 GB | 2XLARGE 824 GB
# The base image is 13.7 GB compressed and ~30 GB extracted, so SMALL is too
# tight. LARGE also brings 8 vCPU, which speeds up the layer extraction.
DEFAULT_COMPUTE = "BUILD_GENERAL1_LARGE"
DEFAULT_IMAGE = "aws/codebuild/standard:7.0"

BUILDSPEC = r"""
version: 0.2
phases:
  pre_build:
    commands:
      - echo "disk before build:" && df -h /
      - aws --version && docker --version
      - aws ecr get-login-password --region "$AWS_DEFAULT_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
      - aws ecr describe-repositories --repository-names "$ECR_REPO" >/dev/null 2>&1 || aws ecr create-repository --repository-name "$ECR_REPO" --image-scanning-configuration scanOnPush=true --encryption-configuration encryptionType=AES256
      - docker pull --platform linux/amd64 "$SGLANG_IMAGE"
  build:
    commands:
      - echo "building $IMAGE_URI"
      - docker build --platform linux/amd64 --build-arg "SGLANG_IMAGE=$SGLANG_IMAGE" --build-arg "SGLANG_COMMIT=$SGLANG_COMMIT" -t "$IMAGE_URI" .
      - echo "disk after build:" && df -h /
  post_build:
    commands:
      - docker push "$IMAGE_URI"
      - echo "pushed $IMAGE_URI"
"""


def ensure_codebuild_role(
    iam,
    *,
    role_name: str,
    account_id: str,
    region: str,
    bucket: str,
    source_key_prefix: str,
    ecr_repo: str,
    project_name: str,
    verbose: bool = True,
) -> str:
    """Create/update a least-privilege CodeBuild service role.

    Scoped the same way as the endpoint role: our ECR repo, the one S3 prefix the
    source zip lives under, and this project's own log group. No wildcards beyond
    the two actions that have no resource-level permissions.
    """
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "codebuild.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {"StringEquals": {"aws:SourceAccount": account_id}},
        }],
    }
    created = False
    try:
        arn = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Builds the MiniMax-H3 serving image and pushes it to ECR",
        )["Role"]["Arn"]
        created = True
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=role_name, PolicyDocument=json.dumps(trust))
        arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]

    log_group = f"arn:aws:logs:{region}:{account_id}:log-group:/aws/codebuild/{project_name}"
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "OwnLogGroupOnly",
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": [log_group, log_group + ":*"],
            },
            {
                "Sid": "ReadSourceZip",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:GetObjectVersion"],
                "Resource": f"arn:aws:s3:::{bucket}/{source_key_prefix.strip('/')}/*",
            },
            {
                "Sid": "PushOurImage",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability", "ecr:CompleteLayerUpload",
                    "ecr:InitiateLayerUpload", "ecr:PutImage", "ecr:UploadLayerPart",
                    "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
                    "ecr:DescribeRepositories", "ecr:CreateRepository",
                ],
                "Resource": f"arn:aws:ecr:{region}:{account_id}:repository/{ecr_repo}",
            },
            {
                # No resource-level permissions exist for these two.
                "Sid": "EcrAuthAndDescribe",
                "Effect": "Allow",
                "Action": ["ecr:GetAuthorizationToken"],
                "Resource": "*",
            },
        ],
    }
    iam.put_role_policy(RoleName=role_name, PolicyName="inline", PolicyDocument=json.dumps(policy))
    if verbose:
        print(f"{'created' if created else 'updated'} CodeBuild role {role_name}")
    if created:
        time.sleep(10)  # IAM eventual consistency
    return arn


def _zip_dir(directory: Path) -> bytes:
    """Zip a directory's contents at the archive root (no wrapping folder)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.name != ".image_uri":
                zf.write(path, path.relative_to(directory).as_posix())
    return buf.getvalue()


def build_image(
    *,
    codebuild,
    s3,
    iam,
    logs,
    account_id: str,
    region: str,
    bucket: str,
    container_dir: Path,
    ecr_repo: str,
    image_tag: str = "latest",
    project_name: str = "minimax-h3-image-build",
    role_name: str = "minimax-h3-codebuild-role",
    source_key_prefix: str = "minimax-h3/build",
    compute_type: str = DEFAULT_COMPUTE,
    sglang_image: str = "lmsysorg/sglang:dev",
    sglang_commit: str = "d257b58e67193780ff8a59ab54b48219b9dc28d2",
    timeout_minutes: int = 120,
    follow: bool = True,
) -> dict[str, Any]:
    """Zip `container/`, run the build in CodeBuild, stream logs, return the image URI."""
    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    image_uri = f"{registry}/{ecr_repo}:{image_tag}"

    role_arn = ensure_codebuild_role(
        iam, role_name=role_name, account_id=account_id, region=region, bucket=bucket,
        source_key_prefix=source_key_prefix, ecr_repo=ecr_repo, project_name=project_name,
    )

    source_key = f"{source_key_prefix.strip('/')}/container-source.zip"
    payload = _zip_dir(container_dir)
    s3.put_object(Bucket=bucket, Key=source_key, Body=payload, ContentType="application/zip")
    print(f"source  s3://{bucket}/{source_key} ({len(payload) / 1024:.0f} kB)")

    env_vars = [
        {"name": "REGISTRY", "value": registry},
        {"name": "ECR_REPO", "value": ecr_repo},
        {"name": "IMAGE_URI", "value": image_uri},
        {"name": "SGLANG_IMAGE", "value": sglang_image},
        {"name": "SGLANG_COMMIT", "value": sglang_commit},
    ]
    project_cfg = {
        "name": project_name,
        "source": {"type": "S3", "location": f"{bucket}/{source_key}", "buildspec": BUILDSPEC},
        "artifacts": {"type": "NO_ARTIFACTS"},
        "environment": {
            "type": "LINUX_CONTAINER",
            "image": DEFAULT_IMAGE,
            "computeType": compute_type,
            "privilegedMode": True,  # required to run a Docker daemon in the build
            "environmentVariables": env_vars,
        },
        "serviceRole": role_arn,
        "timeoutInMinutes": timeout_minutes,
    }
    try:
        codebuild.create_project(**project_cfg)
        print(f"created project {project_name} ({compute_type}, privileged)")
    except codebuild.exceptions.ResourceAlreadyExistsException:
        codebuild.update_project(**project_cfg)
        print(f"updated project {project_name} ({compute_type}, privileged)")

    build_id = codebuild.start_build(projectName=project_name)["build"]["id"]
    print(f"build   {build_id}")
    print(f"console https://{region}.console.aws.amazon.com/codesuite/codebuild/"
          f"{account_id}/projects/{project_name}/build/{build_id.replace(':', '%3A')}")
    if not follow:
        return {"build_id": build_id, "image_uri": image_uri, "status": "IN_PROGRESS"}

    return _follow(codebuild, logs, build_id, image_uri, timeout_minutes)


def _follow(codebuild, logs, build_id: str, image_uri: str, timeout_minutes: int) -> dict[str, Any]:
    """Poll the build, printing phase transitions and new log lines."""
    started = time.time()
    seen_phase = None
    token = None
    group = stream = None

    while time.time() - started < timeout_minutes * 60 + 120:
        build = codebuild.batch_get_builds(ids=[build_id])["builds"][0]
        phase = build.get("currentPhase")
        if phase != seen_phase:
            print(f"[{time.time() - started:6.0f}s] phase: {phase}")
            seen_phase = phase

        info = build.get("logs") or {}
        group, stream = info.get("groupName", group), info.get("streamName", stream)
        if group and stream:
            try:
                kw = {"logGroupName": group, "logStreamName": stream, "startFromHead": True}
                if token:
                    kw["nextToken"] = token
                page = logs.get_log_events(**kw)
                for event in page["events"]:
                    print("  " + event["message"].rstrip())
                if page["events"] or token is None:
                    token = page.get("nextForwardToken")
            except logs.exceptions.ResourceNotFoundException:
                pass

        if build["buildStatus"] != "IN_PROGRESS":
            status = build["buildStatus"]
            print(f"\nbuild {status} after {(time.time() - started) / 60:.1f} min")
            if status != "SUCCEEDED":
                failed = [p for p in build.get("phases", []) if p.get("phaseStatus") == "FAILED"]
                for p in failed:
                    for ctx in p.get("contexts", []):
                        print(f"  {p['phaseType']}: {ctx.get('message')}")
                raise RuntimeError(
                    f"CodeBuild {status}. If it ran out of disk, retry with "
                    f"compute_type='BUILD_GENERAL1_2XLARGE' (824 GB)."
                )
            return {"build_id": build_id, "image_uri": image_uri, "status": status,
                    "minutes": round((time.time() - started) / 60, 1)}
        time.sleep(10)

    raise TimeoutError(f"build {build_id} still running after {timeout_minutes} min")
