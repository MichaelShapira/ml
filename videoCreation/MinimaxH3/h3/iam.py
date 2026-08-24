"""
A dedicated least-privilege execution role for the MiniMax-H3 endpoint.

This follows the pattern established by `whisper_yd` in this workspace, whose
steering is explicit about it: *"Do NOT revert the endpoint to the broad
SageMaker notebook role"*, and *"When adding capability, add a new
narrowly-scoped statement, never a wildcard action/resource."*

The endpoint needs exactly six things:

1. pull **our** image from **our** ECR repo (not a wildcard, not the AWS DLC
   registry - this is a BYOC image),
2. read the staged weights prefix (SageMaker downloads them as this role),
3. read the async request objects and any `s3://` conditions the caller passes,
4. write the generated MP4s, async responses and async failures,
5. write to its own CloudWatch log group,
6. publish its own namespaced metrics.

Nothing else. In particular: no `s3:*`, no `s3:DeleteObject`, no bucket-wide
`GetObject`, and no `AmazonSageMakerFullAccess`.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from typing import Any

__all__ = [
    "endpoint_trust_policy",
    "endpoint_inline_policy",
    "ensure_endpoint_role",
    "diagnose_endpoint_role",
    "delete_endpoint_role",
]


def endpoint_trust_policy(account_id: str, *, restrict_to_account: bool = False) -> dict:
    """Let SageMaker assume the role.

    ``restrict_to_account=True`` adds an ``aws:SourceAccount`` condition. It is
    **off by default and you probably want to leave it off**: SageMaker validates
    the execution role when you create a model or endpoint config, and that
    validation performs a simulated ``AssumeRole`` with no ``aws:SourceAccount``
    in the request context. The condition therefore evaluates false and the whole
    deploy fails with:

        The provided role '...' is invalid. Please ensure the role exists, that
        its trust relationship policy allows the action "sts:AssumeRole" for the
        service principal "sagemaker.amazonaws.com" ...

    What is given up is small. The confused-deputy pattern that ``aws:SourceAccount``
    defends against needs a third party able to induce *your* SageMaker into using
    this role; endpoint configs reference the role ARN from inside this account, and
    only ``sagemaker.amazonaws.com`` can assume it at all. This is the canonical
    SageMaker execution-role trust policy for that reason.
    """
    statement: dict[str, Any] = {
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }
    if restrict_to_account:
        statement["Condition"] = {"StringEquals": {"aws:SourceAccount": account_id}}
    return {"Version": "2012-10-17", "Statement": [statement]}


def endpoint_inline_policy(
    *,
    account_id: str,
    region: str,
    bucket: str,
    weights_prefix: str,
    io_prefix: str,
    ecr_repo: str,
    endpoint_name: str,
) -> dict:
    """Build the inline policy.

    ``weights_prefix`` and ``io_prefix`` are *key* prefixes inside ``bucket``
    (e.g. ``minimax-h3/weights``), not full ``s3://`` URIs.
    """
    weights = weights_prefix.strip("/")
    io = io_prefix.strip("/")
    log_group = f"arn:aws:logs:{region}:{account_id}:log-group:/aws/sagemaker/Endpoints/{endpoint_name}"

    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PullOurImageLayers",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchCheckLayerAvailability",
                ],
                "Resource": f"arn:aws:ecr:{region}:{account_id}:repository/{ecr_repo}",
            },
            {
                # GetAuthorizationToken has no resource-level permissions; "*" is
                # the only valid resource for it. It grants a token, not access.
                "Sid": "EcrAuthToken",
                "Effect": "Allow",
                "Action": ["ecr:GetAuthorizationToken"],
                "Resource": "*",
            },
            {
                "Sid": "ReadStagedWeights",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": f"arn:aws:s3:::{bucket}/{weights}/*",
            },
            {
                # NO s3:prefix Condition, deliberately.
                #
                # SageMaker lists the uncompressed weights prefix to enumerate
                # objects before downloading them, and its role validator also
                # checks s3:ListBucket on the bucket. That check runs with no
                # `s3:prefix` in the request context, so a StringLike condition on
                # s3:prefix evaluates false and the deploy is rejected with
                # "...and has s3:ListBucket ... permissions for bucket X".
                #
                # The cost of dropping it is that the role can enumerate object
                # keys anywhere in this bucket. That is metadata, not data: reads
                # and writes are still confined to the two prefixes below.
                "Sid": "ListBucket",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": f"arn:aws:s3:::{bucket}",
            },
            {
                # The whole io/ prefix rather than requests/ + demo-inputs/ +
                # generated/ separately. Two reasons: the shim reads conditions
                # from `demo-inputs/` and `generated/` (the demo chain feeds a
                # clip's own extracted frames back in), and SageMaker's validator
                # probes object-level access with a key it chooses, which narrow
                # sub-prefixes can fail. weights/ still has its own read-only
                # statement and nothing outside these two prefixes is reachable.
                "Sid": "ReadWriteIoPrefix",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": f"arn:aws:s3:::{bucket}/{io}/*",
            },
            {
                "Sid": "OwnLogGroupOnly",
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": [log_group, log_group + ":*"],
            },
            {
                "Sid": "NamespacedMetrics",
                "Effect": "Allow",
                "Action": ["cloudwatch:PutMetricData"],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "cloudwatch:namespace": ["/aws/sagemaker/Endpoints", "AWS/SageMaker"]
                    }
                },
            },
        ],
    }


def ensure_endpoint_role(
    iam,
    *,
    role_name: str,
    account_id: str,
    region: str,
    bucket: str,
    weights_prefix: str,
    io_prefix: str,
    ecr_repo: str,
    endpoint_name: str,
    restrict_trust_to_account: bool = False,
    settle_s: int = 15,
    verbose: bool = True,
) -> str:
    """Create or update the role and its inline policy. Idempotent.

    Returns the role ARN. Always waits ``settle_s`` afterwards: IAM is eventually
    consistent, and ``create_model`` validating a role whose trust policy was
    updated seconds ago fails with the same "provided role is invalid" message as
    a genuinely wrong policy - which sends you debugging the wrong thing.
    """
    trust = endpoint_trust_policy(account_id, restrict_to_account=restrict_trust_to_account)
    created = False
    try:
        arn = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Least-privilege execution role for the MiniMax-H3 async endpoint",
        )["Role"]["Arn"]
        created = True
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=role_name, PolicyDocument=json.dumps(trust))
        arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]

    policy = endpoint_inline_policy(
        account_id=account_id,
        region=region,
        bucket=bucket,
        weights_prefix=weights_prefix,
        io_prefix=io_prefix,
        ecr_repo=ecr_repo,
        endpoint_name=endpoint_name,
    )
    iam.put_role_policy(RoleName=role_name, PolicyName="inline", PolicyDocument=json.dumps(policy))

    if verbose:
        print(f"{'created' if created else 'updated'} role {role_name}")
        print(f"  {arn}")
        print(f"  trust: sagemaker.amazonaws.com"
              + (f", conditioned on aws:SourceAccount={account_id}" if restrict_trust_to_account
                 else " (unconditional - required by SageMaker's role validator)"))
        for stmt in policy["Statement"]:
            cond = "  [conditional]" if "Condition" in stmt else ""
            print(f"  {stmt['Sid']:<22} {', '.join(stmt['Action'])}{cond}")
    if settle_s:
        if verbose:
            print(f"  waiting {settle_s}s for IAM to settle")
        time.sleep(settle_s)
    return arn


def diagnose_endpoint_role(
    iam,
    *,
    role_name: str,
    account_id: str,
    bucket: str,
    weights_prefix: str,
    io_prefix: str,
    ecr_repo: str,
    region: str,
) -> bool:
    """Simulate the checks SageMaker runs when it validates the execution role.

    ``The provided role '...' is invalid`` names three things it looked at (trust,
    ``s3:ListBucket``, ``s3:PutObject``) without saying which one failed. This
    reproduces each check so you can see it instead of guessing.

    The important subtlety: SageMaker simulates with an **empty request context**.
    Any `Condition` referencing a request key - ``s3:prefix``, ``aws:SourceAccount`` -
    evaluates false there even though it would pass in real use.
    """
    weights, io = weights_prefix.strip("/"), io_prefix.strip("/")
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    ok = True

    try:
        role = iam.get_role(RoleName=role_name)["Role"]
    except iam.exceptions.NoSuchEntityException:
        print(f"FAIL role {role_name} does not exist")
        return False

    trust = role["AssumeRolePolicyDocument"]
    if isinstance(trust, str):
        trust = json.loads(urllib.parse.unquote(trust))
    print(f"role {role_arn}")
    services, conditioned = [], False
    for stmt in trust.get("Statement", []):
        principal = stmt.get("Principal", {})
        service = principal.get("Service")
        services += [service] if isinstance(service, str) else (service or [])
        if "Condition" in stmt:
            conditioned = True
    trust_ok = "sagemaker.amazonaws.com" in services and not conditioned
    ok &= trust_ok
    print(f"{'ok  ' if trust_ok else 'FAIL'} trust: services={services}"
          f"{', CONDITIONAL' if conditioned else ', unconditional'}")
    if conditioned:
        print("     -> a Condition on the trust policy makes SageMaker's simulated AssumeRole")
        print("        fail. Re-create the role with restrict_trust_to_account=False.")

    checks = [
        ("s3:ListBucket", f"arn:aws:s3:::{bucket}", "SageMaker checks this with no s3:prefix context"),
        ("s3:PutObject", f"arn:aws:s3:::{bucket}/{io}/responses/probe", "async response path"),
        ("s3:PutObject", f"arn:aws:s3:::{bucket}/{io}/failures/probe", "async failure path"),
        ("s3:GetObject", f"arn:aws:s3:::{bucket}/{io}/requests/probe.json", "async request path"),
        ("s3:GetObject", f"arn:aws:s3:::{bucket}/{weights}/FL2VA/model_index.json", "weights"),
        ("ecr:BatchGetImage", f"arn:aws:ecr:{region}:{account_id}:repository/{ecr_repo}", "image pull"),
    ]
    for action, resource, note in checks:
        try:
            results = iam.simulate_principal_policy(
                PolicySourceArn=role_arn, ActionNames=[action], ResourceArns=[resource]
            )["EvaluationResults"]
            decision = results[0]["EvalDecision"]
        except Exception as exc:  # noqa: BLE001
            print(f"?    {action:<20} could not simulate ({type(exc).__name__})")
            continue
        good = decision == "allowed"
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {action:<20} {decision:<16} {note}")
        if not good:
            print(f"     resource: {resource}")

    # Anything the endpoint should NOT be able to do.
    for action, resource in [
        ("s3:DeleteObject", f"arn:aws:s3:::{bucket}/{io}/generated/x.mp4"),
        ("s3:GetObject", f"arn:aws:s3:::{bucket}/some-other-prefix/secret"),
        ("sagemaker:CreateEndpoint", "*"),
    ]:
        try:
            decision = iam.simulate_principal_policy(
                PolicySourceArn=role_arn, ActionNames=[action], ResourceArns=[resource]
            )["EvaluationResults"][0]["EvalDecision"]
            denied = decision != "allowed"
            ok &= denied
            print(f"{'ok  ' if denied else 'FAIL'} {action:<20} {decision:<16} must be denied")
        except Exception:  # noqa: BLE001
            pass

    print("\nrole looks valid to SageMaker" if ok else "\nrole would be REJECTED - see FAIL above")
    return ok


def delete_endpoint_role(iam, role_name: str, *, verbose: bool = True) -> None:
    """Remove the inline policy and the role, tolerating either being absent."""
    for call, kwargs in (
        (iam.delete_role_policy, {"RoleName": role_name, "PolicyName": "inline"}),
        (iam.delete_role, {"RoleName": role_name}),
    ):
        try:
            call(**kwargs)
        except iam.exceptions.NoSuchEntityException:
            continue
    if verbose:
        print(f"deleted role {role_name}")
