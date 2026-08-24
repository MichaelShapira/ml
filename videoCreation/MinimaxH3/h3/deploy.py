"""
Deployment helpers for the MiniMax-H3 async endpoint.

The fiddly parts this encapsulates, all of them learned the hard way on
SageMaker BYOC deployments of large diffusion models:

* **Delete before recreate.** Pointing an existing endpoint name at a new model
  or config does not reliably replace a crash-looping container. Deleting the
  endpoint *and* its config first is the only dependable path.
* **Uncompressed model data.** Weights are staged as plain S3 objects under a
  prefix (``CompressionType: None``), not a tarball. SageMaker mirrors the prefix
  into ``/opt/ml/model`` verbatim, so iterating on the container never means
  re-archiving 288 GB.
* **Generous timeouts.** ``ModelDataDownloadTimeoutInSeconds`` and
  ``ContainerStartupHealthCheckTimeoutInSeconds`` both go to the 3600 s maximum:
  the download alone is 288 GB and two 4-GPU model loads follow it.
* **No VolumeSizeInGB.** ml.p5.48xlarge has local NVMe, and SageMaker rejects an
  explicit EBS volume size for instance families that do.
"""

from __future__ import annotations

import time
from typing import Any

__all__ = ["EndpointSpec", "deploy", "wait_for_endpoint", "delete_endpoint", "describe"]

MAX_TIMEOUT_S = 3600


class EndpointSpec:
    """Everything needed to stand up the endpoint, in one place."""

    def __init__(
        self,
        *,
        name: str,
        image_uri: str,
        role_arn: str,
        async_output_s3_prefix: str,
        model_data_s3_prefix: str | None = None,
        instance_type: str = "ml.p5.48xlarge",
        variants: str = "fl2va,ref2va",
        gpus_per_backend: int = 4,
        tp_size: int = 4,
        ulysses_degree: int = 1,
        performance_mode: str = "speed",
        warmup_on_start: bool = True,
        generated_s3_prefix: str | None = None,
        sns_success_topic_arn: str | None = None,
        sns_error_topic_arn: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.image_uri = image_uri
        self.role_arn = role_arn
        self.async_output_s3_prefix = async_output_s3_prefix.rstrip("/")
        self.model_data_s3_prefix = model_data_s3_prefix
        self.instance_type = instance_type
        self.variants = variants
        self.gpus_per_backend = gpus_per_backend
        self.tp_size = tp_size
        self.ulysses_degree = ulysses_degree
        self.performance_mode = performance_mode
        self.warmup_on_start = warmup_on_start
        self.generated_s3_prefix = generated_s3_prefix
        self.sns_success_topic_arn = sns_success_topic_arn
        self.sns_error_topic_arn = sns_error_topic_arn
        self.extra_env = extra_env or {}

    def async_inference_config(self, max_concurrent_invocations: int) -> dict[str, Any]:
        """Build ``AsyncInferenceConfig``.

        SNS notification is optional but recommended: a generation takes minutes,
        so a completion event beats polling S3. Note the deliberate omission of
        ``IncludeInferenceResponseIn`` - our response carries S3 URIs and job
        metadata, and there is no reason to fan that out to a topic. If you set
        ``H3_OUTPUT_S3_PREFIX`` to nothing, the response contains base64 video
        and must NOT be included in a notification.
        """
        output: dict[str, Any] = {
            "S3OutputPath": f"{self.async_output_s3_prefix}/responses/",
            "S3FailurePath": f"{self.async_output_s3_prefix}/failures/",
        }
        if self.sns_success_topic_arn or self.sns_error_topic_arn:
            notification: dict[str, Any] = {}
            if self.sns_success_topic_arn:
                notification["SuccessTopic"] = self.sns_success_topic_arn
            if self.sns_error_topic_arn:
                notification["ErrorTopic"] = self.sns_error_topic_arn
            output["NotificationConfig"] = notification
        return {
            "OutputConfig": output,
            "ClientConfig": {"MaxConcurrentInvocationsPerInstance": max_concurrent_invocations},
        }

    def environment(self) -> dict[str, str]:
        env = {
            "H3_VARIANTS": self.variants,
            "H3_GPUS_PER_BACKEND": str(self.gpus_per_backend),
            "H3_TP_SIZE": str(self.tp_size),
            "H3_ULYSSES_DEGREE": str(self.ulysses_degree),
            "H3_PERFORMANCE_MODE": self.performance_mode,
            "H3_WARMUP_ON_START": "1" if self.warmup_on_start else "0",
            # Bound below SageMaker's 3600 s invocation ceiling so a stuck job
            # returns a readable error instead of being cut off mid-response.
            "H3_JOB_TIMEOUT_S": "3300",
            "H3_BACKEND_BOOT_TIMEOUT_S": "3000",
        }
        if self.generated_s3_prefix:
            env["H3_OUTPUT_S3_PREFIX"] = self.generated_s3_prefix.rstrip("/")
        env.update(self.extra_env)
        return env

    def container(self) -> dict[str, Any]:
        container: dict[str, Any] = {"Image": self.image_uri, "Environment": self.environment()}
        if self.model_data_s3_prefix:
            prefix = self.model_data_s3_prefix.rstrip("/") + "/"
            container["ModelDataSource"] = {
                "S3DataSource": {
                    "S3Uri": prefix,
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            }
        return container

    def describe(self) -> str:
        gpus = self.gpus_per_backend * len(self.variants.split(","))
        weights = self.model_data_s3_prefix or "(downloaded from the Hub at first boot)"
        sns = self.sns_success_topic_arn or self.sns_error_topic_arn or "(none - poll S3)"
        return (
            f"endpoint       {self.name}\n"
            f"instance       {self.instance_type}\n"
            f"image          {self.image_uri}\n"
            f"role           {self.role_arn}\n"
            f"weights        {weights}\n"
            f"variants       {self.variants}  ({gpus} GPUs total, "
            f"{self.gpus_per_backend}/backend, tp={self.tp_size} ulysses={self.ulysses_degree})\n"
            f"async output   {self.async_output_s3_prefix}\n"
            f"notification   {sns}\n"
            f"warmup on boot {self.warmup_on_start}"
        )


def describe(sm, name: str) -> dict[str, Any] | None:
    try:
        return sm.describe_endpoint(EndpointName=name)
    except sm.exceptions.ClientError:
        return None


def delete_endpoint(sm, name: str, *, wait: bool = True, verbose: bool = True) -> None:
    """Remove the endpoint and its config, tolerating anything already gone."""
    for call, kwargs, label in (
        (sm.delete_endpoint, {"EndpointName": name}, "endpoint"),
        (sm.delete_endpoint_config, {"EndpointConfigName": name}, "endpoint config"),
        (sm.delete_model, {"ModelName": name}, "model"),
    ):
        try:
            call(**kwargs)
            if verbose:
                print(f"deleted {label} {name}")
        except sm.exceptions.ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            message = exc.response.get("Error", {}).get("Message", "")
            if "Could not find" in message or code in ("ValidationException", "ResourceNotFound"):
                continue
            raise
    if wait:
        deadline = time.time() + 900
        while time.time() < deadline:
            if describe(sm, name) is None:
                return
            time.sleep(10)


def deploy(
    sm,
    spec: EndpointSpec,
    *,
    replace: bool = True,
    max_concurrent_invocations: int = 1,
    verbose: bool = True,
) -> str:
    """Create the model, async endpoint config and endpoint. Returns the ARN."""
    if verbose:
        print(spec.describe())
        print()

    if replace and describe(sm, spec.name) is not None:
        if verbose:
            print(f"{spec.name} exists - deleting first (recreating in place is unreliable)")
        delete_endpoint(sm, spec.name, verbose=verbose)
    else:
        # Even without an endpoint, a leftover model/config with this name blocks
        # creation. Clear them without waiting.
        delete_endpoint(sm, spec.name, wait=False, verbose=False)

    # SageMaker validates the execution role here, and a role whose trust policy
    # was written seconds ago fails validation with exactly the same message as a
    # genuinely wrong one ("The provided role '...' is invalid"). Retry a few
    # times before believing it, then say which it probably was.
    for attempt in range(1, 5):
        try:
            sm.create_model(
                ModelName=spec.name,
                ExecutionRoleArn=spec.role_arn,
                PrimaryContainer=spec.container(),
                EnableNetworkIsolation=False,  # the container may fetch weights from the Hub
            )
            break
        except Exception as exc:  # noqa: BLE001 - botocore error classes vary
            message = str(exc)
            transient = "provided role" in message and "invalid" in message
            if not transient or attempt == 4:
                if transient:
                    raise RuntimeError(
                        f"{message}\n\n"
                        "This persisted across retries, so it is the policy rather than IAM "
                        "propagation. Run h3.iam.diagnose_endpoint_role() - it simulates each "
                        "check SageMaker makes. The usual cause is a Condition that SageMaker's "
                        "validator cannot satisfy: it simulates with an empty request context, so "
                        "a Condition on aws:SourceAccount (trust) or s3:prefix (ListBucket) "
                        "evaluates false."
                    ) from exc
                raise
            wait = 10 * attempt
            if verbose:
                print(f"role not accepted yet (attempt {attempt}/4), waiting {wait}s - "
                      "usually IAM propagation")
            time.sleep(wait)

    sm.create_endpoint_config(
        EndpointConfigName=spec.name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": spec.name,
                "InitialInstanceCount": 1,
                "InstanceType": spec.instance_type,
                "ModelDataDownloadTimeoutInSeconds": MAX_TIMEOUT_S,
                "ContainerStartupHealthCheckTimeoutInSeconds": MAX_TIMEOUT_S,
            }
        ],
        AsyncInferenceConfig=spec.async_inference_config(max_concurrent_invocations),
    )

    response = sm.create_endpoint(EndpointName=spec.name, EndpointConfigName=spec.name)
    if verbose:
        print(f"creating endpoint {spec.name} - expect 20-40 min "
              f"(image pull, weight download, then two model loads + warmup)")
    return response["EndpointArn"]


def wait_for_endpoint(
    sm,
    logs,
    name: str,
    *,
    timeout_s: int = 4800,
    poll_s: int = 30,
    show_logs: bool = True,
) -> str:
    """Poll until InService or Failed, printing progress from CloudWatch.

    ``Creating`` means it is still working. ``Failed`` plus ``FailureReason`` is
    the real answer. No log group at all means the container never started, so
    the problem is upstream of your code.
    """
    group = f"/aws/sagemaker/Endpoints/{name}"
    started = time.time()
    seen: set[str] = set()
    last_status = ""

    while time.time() - started < timeout_s:
        desc = sm.describe_endpoint(EndpointName=name)
        status = desc["EndpointStatus"]
        if status != last_status:
            print(f"[{time.time() - started:6.0f}s] status={status}")
            last_status = status

        if show_logs:
            try:
                events = logs.filter_log_events(
                    logGroupName=group,
                    startTime=int((started - 120) * 1000),
                    filterPattern='?"[shim" ?READY ?"BOOT FAILED" ?Error ?error ?Traceback',
                    limit=40,
                ).get("events", [])
                for event in events:
                    key = f"{event['timestamp']}:{event['message'][:80]}"
                    if key not in seen:
                        seen.add(key)
                        print("   " + event["message"].rstrip())
            except logs.exceptions.ResourceNotFoundException:
                pass

        if status == "InService":
            print(f"InService after {(time.time() - started) / 60:.1f} min")
            return status
        if status in ("Failed", "OutOfService", "RollingBack"):
            print(f"FAILED: {desc.get('FailureReason')}")
            return status
        time.sleep(poll_s)

    raise TimeoutError(f"endpoint {name} still {last_status} after {timeout_s}s")
