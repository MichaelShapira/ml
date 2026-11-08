"""SageMaker helpers for the FLUX 3 Image Klein g7e notebook (boto3 only; no SageMaker Python SDK needed).

The notebook calls these so each cell stays short and the same code can be exercised from a script.

    create_hf_token_secret() / delete_secret()   the token lives in AWS only while the staging job runs
    stage_weights_job()   Hub -> S3, as a CPU SageMaker Processing job (Ray Serve DLC image)
    upload_serving()      config.yaml + code/ into the model prefix (seconds; never touches the weights)
    deploy()              async endpoint on ml.g7e.2xlarge; retries when SageMaker has no capacity
    wait_ready()          InService is not enough: waits for the app's READY line in CloudWatch
    invoke()              S3 request -> invoke_endpoint_async -> poll output/failure
    set_auto_stop()       server-side guard: scale the endpoint to 0 instances at a given time
    delete()              endpoint, endpoint config, model, autoscaling target
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent

# AWS Ray Serve DLC for SageMaker (GPU): Ray 2.59, PyTorch 2.10 + CUDA 12.9, Transformers 5.19, Python 3.13.
# Pinned to a patch tag so code/requirements.txt's guard pins stay true. 763104351884 hosts the DLCs in most
# commercial regions; see https://aws.github.io/deep-learning-containers/ray/ for others.
RAY_DLC_TAG = "serve-ml-sagemaker-cuda-v1.5.7"
DLC_ACCOUNT = "763104351884"
# Driver 580 / CUDA 13.0 host AMI; the Ray DLC docs require an explicit inference AMI for GPU endpoints.
INFERENCE_AMI = "al2023-ami-sagemaker-inference-gpu-4-1"
CAPACITY_MARKERS = ("capacity", "insufficient", "unable to provision")


def log(message: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {message}", flush=True)


@dataclass
class Settings:
    region: str
    bucket: str
    role_arn: str
    project: str = "flux3-klein-g7e"
    endpoint_name: str = "flux3-klein-g7e"
    instance_type: str = "ml.g7e.2xlarge"
    image_uri: str = ""
    inference_ami: str = INFERENCE_AMI
    model_dir: str = "model"  # S3 prefix (under project) mirrored to /opt/ml/model
    extra_env: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.image_uri:
            self.image_uri = f"{DLC_ACCOUNT}.dkr.ecr.{self.region}.amazonaws.com/ray:{RAY_DLC_TAG}"

    @property
    def model_prefix(self) -> str:
        return f"{self.project}/{self.model_dir}"

    @property
    def model_uri(self) -> str:
        return f"s3://{self.bucket}/{self.model_prefix}/"

    @property
    def weights_prefix(self) -> str:
        return f"{self.model_prefix}/weights"

    @property
    def io_prefix(self) -> str:
        return f"{self.project}/io"

    @property
    def log_group(self) -> str:
        return f"/aws/sagemaker/Endpoints/{self.endpoint_name}"

    def client(self, service: str):
        return boto3.client(service, region_name=self.region)


# ----------------------------------------------------------------------------- staging
def create_hf_token_secret(cfg: Settings, token: str) -> str:
    """Put the Hugging Face token in a short-lived Secrets Manager secret; returns its name.

    The staging job reads it from there, so the token never appears in the job's description (which SageMaker
    keeps after the job ends). The `AmazonSageMaker-` prefix is what the managed AmazonSageMakerFullAccess
    policy lets a SageMaker execution role read, so no extra IAM statement is needed. The name is unique per
    run; call delete_secret() as soon as the job finishes (the notebook does, in a `finally`).
    """
    name = f"AmazonSageMaker-flux3-hf-token-{dt.datetime.utcnow():%Y%m%d%H%M%S}"
    cfg.client("secretsmanager").create_secret(
        Name=name, SecretString=token, Tags=[{"Key": "SageMaker", "Value": "true"}],
        Description="Temporary Hugging Face read token for one FLUX 3 Image Klein staging job")
    log(f"created temporary secret {name}")
    return name


def delete_secret(cfg: Settings, name: str) -> None:
    """Delete the secret immediately (no recovery window)."""
    try:
        cfg.client("secretsmanager").delete_secret(SecretId=name, ForceDeleteWithoutRecovery=True)
        log(f"deleted secret {name}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise


def weights_staged(cfg: Settings) -> dict | None:
    """The staging manifest if staging completed, else None."""
    try:
        body = cfg.client("s3").get_object(Bucket=cfg.bucket, Key=f"{cfg.weights_prefix}/_flux3_stage.json")["Body"]
        return json.loads(body.read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def stage_weights_job(cfg: Settings, hf_token_secret: str | None, repo: str = "black-forest-labs/flux-3-image-klein",
                      instance_type: str = "ml.c6i.4xlarge", wait: bool = True) -> str:
    """Run scripts/stage_weights.py as a SageMaker Processing job. Returns the job name."""
    s3, sm = cfg.client("s3"), cfg.client("sagemaker")
    code_key = f"{cfg.project}/staging/code/stage_weights.py"
    s3.upload_file(str(HERE / "scripts" / "stage_weights.py"), cfg.bucket, code_key)
    job = f"flux3-stage-{dt.datetime.utcnow():%Y%m%d-%H%M%S}"
    args = ["--repo", repo, "--bucket", cfg.bucket, "--prefix", cfg.weights_prefix, "--region", cfg.region,
            "--scratch", "/opt/ml/processing/scratch"]
    if hf_token_secret:
        args += ["--hf-token-secret", hf_token_secret]
    sm.create_processing_job(
        ProcessingJobName=job,
        RoleArn=cfg.role_arn,
        AppSpecification={
            "ImageUri": cfg.image_uri,
            "ContainerEntrypoint": ["python", "/opt/ml/processing/input/code/stage_weights.py"],
            "ContainerArguments": args,
        },
        ProcessingInputs=[{
            "InputName": "code",
            "S3Input": {"S3Uri": f"s3://{cfg.bucket}/{code_key}", "LocalPath": "/opt/ml/processing/input/code",
                        "S3DataType": "S3Prefix", "S3InputMode": "File"},
        }],
        ProcessingResources={"ClusterConfig": {"InstanceCount": 1, "InstanceType": instance_type,
                                               "VolumeSizeInGB": 120}},
        StoppingCondition={"MaxRuntimeInSeconds": 7200},
        Environment={"AWS_REGION": cfg.region, "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"},
    )
    log(f"processing job {job} started on {instance_type}")
    if wait:
        wait_processing_job(cfg, job)
    return job


def wait_processing_job(cfg: Settings, job: str, poll_s: int = 20) -> None:
    sm, logs = cfg.client("sagemaker"), cfg.client("logs")
    seen, last = set(), None
    while True:
        desc = sm.describe_processing_job(ProcessingJobName=job)
        status = desc["ProcessingJobStatus"]
        if status != last:
            log(f"{job}: {status}")
            last = status
        try:
            # Filter in Python: a CloudWatch pattern starting with "[" is parsed as a space-delimited pattern.
            for page in logs.get_paginator("filter_log_events").paginate(
                    logGroupName="/aws/sagemaker/ProcessingJobs", logStreamNamePrefix=job):
                for event in page["events"]:
                    if event["eventId"] not in seen and event["message"].startswith("[stage"):
                        seen.add(event["eventId"])
                        print("   ", event["message"].rstrip(), flush=True)
        except ClientError:
            pass
        if status in ("Completed", "Failed", "Stopped"):
            if status != "Completed":
                raise RuntimeError(f"{job} {status}: {desc.get('FailureReason') or desc.get('ExitMessage')}")
            return
        time.sleep(poll_s)


# ----------------------------------------------------------------------------- serving code
def upload_serving(cfg: Settings) -> None:
    """config.yaml -> <model>/config.yaml, serving/code/* -> <model>/code/. A few KB; weights untouched."""
    s3 = cfg.client("s3")
    serving = HERE / "serving"
    s3.upload_file(str(serving / "config.yaml"), cfg.bucket, f"{cfg.model_prefix}/config.yaml")
    for path in sorted((serving / "code").iterdir()):
        if path.is_file() and not path.name.startswith("."):
            s3.upload_file(str(path), cfg.bucket, f"{cfg.model_prefix}/code/{path.name}")
    keys = [o["Key"][len(cfg.model_prefix) + 1:] for o in s3.list_objects_v2(
        Bucket=cfg.bucket, Prefix=cfg.model_prefix + "/")["Contents"] if not o["Key"].startswith(cfg.weights_prefix)]
    log(f"serving files in {cfg.model_uri}: {keys}")


# ----------------------------------------------------------------------------- endpoint lifecycle
def _model_environment(cfg: Settings) -> dict:
    env = {
        "FLUX3_WEIGHTS_DIR": "/opt/ml/model/weights",
        "RAYSERVE_BACKEND_TIMEOUT": "3600",  # adapter -> Ray Serve proxy timeout (default 300 s)
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/tmp/hf",
        "TRITON_CACHE_DIR": "/tmp/triton",
        "RAY_USAGE_STATS_ENABLED": "0",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    env.update(cfg.extra_env)
    return env


def delete(cfg: Settings, wait: bool = True) -> None:
    """Delete the endpoint, its config, the model and the autoscaling target; absent pieces are fine."""
    sm, aas = cfg.client("sagemaker"), cfg.client("application-autoscaling")
    name = cfg.endpoint_name
    try:
        aas.deregister_scalable_target(ServiceNamespace="sagemaker", ResourceId=f"endpoint/{name}/variant/AllTraffic",
                                       ScalableDimension="sagemaker:variant:DesiredInstanceCount")
        log(f"deregistered autoscaling target for {name}")
    except ClientError:
        pass
    for call, kwargs in ((sm.delete_endpoint, {"EndpointName": name}),
                         (sm.delete_endpoint_config, {"EndpointConfigName": name}),
                         (sm.delete_model, {"ModelName": name})):
        deadline = time.time() + 3600
        while True:
            try:
                call(**kwargs)
                log(f"{call.__name__}({name}) in {cfg.region}")
                break
            except ClientError as exc:
                text = str(exc)
                if "Could not find" in text or "does not exist" in text:
                    break
                # SageMaker refuses to delete an endpoint while it is Creating/Updating (a capacity attempt can
                # hold it there ~30 min): wait for it to settle, then delete.
                if "in-progress" in text and time.time() < deadline:
                    log(f"{cfg.region}/{name} is still being created; deleting it once it settles")
                    time.sleep(30)
                    continue
                raise
    if wait:
        for _ in range(90):
            try:
                sm.describe_endpoint(EndpointName=name)
            except ClientError:
                log(f"endpoint {name} is gone")
                return
            time.sleep(10)
        raise TimeoutError(f"endpoint {name} still exists after 15 minutes")


def _create(cfg: Settings) -> None:
    sm = cfg.client("sagemaker")
    name = cfg.endpoint_name
    for attempt in range(5):  # a fresh role/permission change can take a moment to be accepted
        try:
            sm.create_model(
                ModelName=name,
                ExecutionRoleArn=cfg.role_arn,
                PrimaryContainer={
                    "Image": cfg.image_uri,
                    "ModelDataSource": {"S3DataSource": {"S3Uri": cfg.model_uri, "S3DataType": "S3Prefix",
                                                         "CompressionType": "None"}},
                    "Environment": _model_environment(cfg),
                },
            )
            break
        except ClientError as exc:
            if "role" in str(exc).lower() and attempt < 4:
                time.sleep(15 * (attempt + 1))
                continue
            raise
    sm.create_endpoint_config(
        EndpointConfigName=name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": name,
            "InstanceType": cfg.instance_type,
            "InitialInstanceCount": 1,
            "ModelDataDownloadTimeoutInSeconds": 3600,
            "ContainerStartupHealthCheckTimeoutInSeconds": 3600,
            "InferenceAmiVersion": cfg.inference_ami,
        }],
        AsyncInferenceConfig={
            "OutputConfig": {"S3OutputPath": f"s3://{cfg.bucket}/{cfg.io_prefix}/outputs/",
                             "S3FailurePath": f"s3://{cfg.bucket}/{cfg.io_prefix}/failures/"},
            "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 1},
        },
    )
    sm.create_endpoint(EndpointName=name, EndpointConfigName=name)


class Cancelled(RuntimeError):
    """deploy() was told to stop through its `cancel` event."""


def deploy(cfg: Settings, retries: int = 3, backoff_s: tuple = (120, 300, 600, 900), poll_s: int = 30,
           reuse: bool = False, cancel=None) -> dict:
    """Create the endpoint and wait for InService. Capacity failures are retried `retries` times (>= 3).

    Delete-before-create: re-pointing an existing endpoint does not reliably replace a broken container.
    Any other failure stops immediately with SageMaker's FailureReason (retrying a code bug wastes ~15 min).
    reuse=True keeps an endpoint of this name that is already InService on the same image and model data
    (g7e capacity is scarce; a kernel restart should not cost the instance). Use reuse=False after editing
    serving/ so the new code is loaded.
    cancel: optional threading.Event; once set, deploy() deletes what it created and raises Cancelled (used when
    racing several regions for capacity).
    """
    def pause(seconds: float) -> None:
        if cancel is not None and cancel.wait(seconds):
            delete(cfg)
            raise Cancelled(f"{cfg.region}: cancelled")
        if cancel is None:
            time.sleep(seconds)

    if reuse:
        existing = _in_service_match(cfg)
        if existing:
            log(f"reusing {cfg.endpoint_name}: InService on {cfg.instance_type} since {existing['CreationTime']:%H:%M}")
            return existing
    retries = max(3, retries)
    for attempt in range(retries + 1):
        delete(cfg)
        if cancel is not None and cancel.is_set():
            raise Cancelled(f"{cfg.region}: cancelled")
        log(f"creating {cfg.endpoint_name} on {cfg.instance_type} in {cfg.region} "
            f"(attempt {attempt + 1}/{retries + 1})")
        try:
            _create(cfg)
        except ClientError as exc:
            text = str(exc)
            if any(m in text.lower() for m in CAPACITY_MARKERS) and attempt < retries:
                wait = backoff_s[min(attempt, len(backoff_s) - 1)]
                log(f"no capacity at create time ({text[:160]}); retrying in {wait}s")
                pause(wait)
                continue
            raise
        desc = wait_in_service(cfg, poll_s, cancel=cancel)
        if desc["EndpointStatus"] == "InService":
            return desc
        reason = desc.get("FailureReason", "")
        if any(m in reason.lower() for m in CAPACITY_MARKERS) and attempt < retries:
            wait = backoff_s[min(attempt, len(backoff_s) - 1)]
            log(f"{cfg.region} capacity failure: {reason[:120]}; retrying in {wait}s")
            pause(wait)
            continue
        raise RuntimeError(f"{cfg.endpoint_name} failed: {reason}")
    raise RuntimeError("unreachable")


def _in_service_match(cfg: Settings) -> dict | None:
    sm = cfg.client("sagemaker")
    try:
        desc = sm.describe_endpoint(EndpointName=cfg.endpoint_name)
        if desc["EndpointStatus"] != "InService":
            return None
        config = sm.describe_endpoint_config(EndpointConfigName=desc["EndpointConfigName"])
        model = sm.describe_model(ModelName=config["ProductionVariants"][0]["ModelName"])
    except ClientError:
        return None
    container = model["PrimaryContainer"]
    same = (container.get("Image") == cfg.image_uri
            and container.get("ModelDataSource", {}).get("S3DataSource", {}).get("S3Uri") == cfg.model_uri
            and config["ProductionVariants"][0]["InstanceType"] == cfg.instance_type)
    return desc if same else None


def wait_in_service(cfg: Settings, poll_s: int = 30, cancel=None) -> dict:
    sm = cfg.client("sagemaker")
    started, last = time.time(), None
    while True:
        desc = sm.describe_endpoint(EndpointName=cfg.endpoint_name)
        status = desc["EndpointStatus"]
        if status != last:
            log(f"{cfg.region}/{cfg.endpoint_name}: {status} (+{time.time() - started:.0f}s)")
            last = status
        if status in ("InService", "Failed", "OutOfService"):
            if status != "InService":
                log(f"{cfg.region} FailureReason: {desc.get('FailureReason')}")
            return desc
        if cancel is not None and cancel.wait(poll_s):
            delete(cfg)
            raise Cancelled(f"{cfg.region}: cancelled while {status}")
        if cancel is None:
            time.sleep(poll_s)


def app_events(cfg: Settings, since_ms: int | None = None, contains: str = "[flux3]") -> list[str]:
    """Container log lines containing `contains` (all lines when empty), oldest first.

    CloudWatch parses a filter pattern that starts with "[" as a space-delimited pattern, so "[flux3]" cannot be
    matched server-side; filter on its letters there and on the exact text here."""
    logs = cfg.client("logs")
    word = "".join(ch for ch in contains if ch.isalnum() or ch == "_")
    kwargs = {"logGroupName": cfg.log_group}
    if word:
        kwargs["filterPattern"] = f'"{word}"'
    if since_ms:
        kwargs["startTime"] = since_ms
    messages = []
    try:
        for page in logs.get_paginator("filter_log_events").paginate(**kwargs):
            messages += [e["message"].rstrip() for e in page["events"] if contains in e["message"]]
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
    return messages


def wait_ready(cfg: Settings, timeout_s: int = 3600, poll_s: int = 20, probe_after_s: int = 900) -> str:
    """The Ray Serve DLC's /ping turns healthy once Serve has routes, before the model finishes loading, so
    InService alone does not mean ready. Wait for the app's own READY (or FATAL) line in CloudWatch, printing the
    load milestones on the way. If no READY line has shown up `probe_after_s` after the endpoint went InService
    (log forwarding is best effort), send one small real request instead: it queues until the model is loaded,
    and a generated image is the strongest readiness signal there is."""
    sm = cfg.client("sagemaker")
    desc = sm.describe_endpoint(EndpointName=cfg.endpoint_name)
    since = int(desc["CreationTime"].timestamp() * 1000)  # only this endpoint's lines; the log group is reused
    in_service_at = desc["LastModifiedTime"].timestamp()
    started, printed = time.time(), set()
    while time.time() - started < timeout_s:
        for line in app_events(cfg, since):
            key = line[-160:]
            if key not in printed:
                printed.add(key)
                print("   ", line[line.find("[flux3]"):][:400], flush=True)
            if "[flux3] FATAL" in line:
                raise RuntimeError("the model failed to load; see the FATAL traceback above")
            if "[flux3] READY" in line:
                log(f"model ready (+{time.time() - started:.0f}s)")
                return line
        if time.time() - in_service_at > probe_after_s:
            log("no READY line in CloudWatch yet; sending a small probe request (it waits for the model)")
            result = invoke(cfg, {"prompt": "a red apple on a wooden table", "upsample": False, "height": 512,
                                  "width": 512, "num_inference_steps": 4, "seed": 0},
                            timeout_s=max(600, int(timeout_s - (time.time() - started))))
            log(f"model ready: probe image {result.image.size} in {result.seconds:.1f}s")
            return "probe"
        time.sleep(poll_s)
    raise TimeoutError("the model did not become ready")


def set_auto_stop(cfg: Settings, hours: float) -> dt.datetime:
    """Server-side safety net: at now+hours, Application Auto Scaling sets the variant to 0 instances.

    Works even if the notebook (or its credentials) is gone. The endpoint stays, with no instance billed;
    run delete() to remove it, or deploy() again to start fresh."""
    aas = cfg.client("application-autoscaling")
    resource = f"endpoint/{cfg.endpoint_name}/variant/AllTraffic"
    dim = "sagemaker:variant:DesiredInstanceCount"
    aas.register_scalable_target(ServiceNamespace="sagemaker", ResourceId=resource, ScalableDimension=dim,
                                 MinCapacity=0, MaxCapacity=1)
    at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0) + dt.timedelta(hours=hours)
    aas.put_scheduled_action(ServiceNamespace="sagemaker", ScheduledActionName=f"{cfg.endpoint_name}-auto-stop",
                             ResourceId=resource, ScalableDimension=dim,
                             Schedule=f"at({at:%Y-%m-%dT%H:%M:%S})",
                             ScalableTargetAction={"MinCapacity": 0, "MaxCapacity": 0})
    log(f"auto-stop scheduled: {cfg.endpoint_name} scales to 0 instances at {at:%Y-%m-%d %H:%M} UTC")
    return at


# ----------------------------------------------------------------------------- invocation
def encode_image(image, fmt: str = "PNG") -> str:
    """PIL image or path -> base64 string for the request's `images` list."""
    from PIL import Image

    if not hasattr(image, "save"):
        image = Image.open(image)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class Result:
    request_id: str
    seconds: float
    body: bytes
    meta: dict

    @property
    def image(self):
        from PIL import Image

        return Image.open(io.BytesIO(self.body))

    @property
    def caption(self) -> str | None:
        return self.meta.get("caption")


def invoke(cfg: Settings, payload: dict, timeout_s: int = 3600, poll_s: float = 1.0) -> Result:
    """Async invoke: request JSON -> S3 -> InvokeEndpointAsync -> poll the output (or failure) object."""
    s3, rt = cfg.client("s3"), cfg.client("sagemaker-runtime")
    request_id = f"{dt.datetime.utcnow():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    key = f"{cfg.io_prefix}/requests/{request_id}.json"
    s3.put_object(Bucket=cfg.bucket, Key=key, Body=json.dumps(payload).encode(), ContentType="application/json")
    started = time.time()
    resp = rt.invoke_endpoint_async(
        EndpointName=cfg.endpoint_name, InputLocation=f"s3://{cfg.bucket}/{key}", ContentType="application/json",
        Accept="application/json" if payload.get("response_format") == "json" else "image/png",
        InvocationTimeoutSeconds=timeout_s, InferenceId=request_id,
    )

    def split(uri: str) -> tuple[str, str]:
        bucket, _, obj = uri.replace("s3://", "", 1).partition("/")
        return bucket, obj

    out_bucket, out_key = split(resp["OutputLocation"])
    fail_bucket, fail_key = split(resp.get("FailureLocation", ""))
    while time.time() - started < timeout_s + 120:
        for bucket, obj, failed in ((out_bucket, out_key, False), (fail_bucket, fail_key, True)):
            if not obj:
                continue
            try:
                body = s3.get_object(Bucket=bucket, Key=obj)["Body"].read()
            except ClientError as exc:
                if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                    continue
                raise
            if failed:
                raise RuntimeError(f"request {request_id} failed: {body[:2000].decode(errors='replace')}")
            seconds = time.time() - started
            if payload.get("response_format") == "json":
                meta = json.loads(body)
                return Result(request_id, seconds, base64.b64decode(meta.pop("image")), meta)
            return Result(request_id, seconds, body, {})
        time.sleep(poll_s)
    raise TimeoutError(f"request {request_id}: no output after {timeout_s}s")
