#!/usr/bin/env bash
#
# AI Photo Booth — one-shot deploy / install script.
#
# Does everything end-to-end so nothing has to be wired by hand:
#   1. builds the SPA (ui/) and the CDK app (cdk/)
#   2. deploys the AiPhotoBoothStack with all parameters + context
#   3. reads the stack outputs (incl. the Invoke_Proxy Function URL)
#   4. regenerates ui/.env.local from those outputs (so `npm run dev` works
#      against the freshly-deployed resources — including invokeFunctionUrl)
#   5. provisions the FLUX 3 Image Klein endpoint config for ml.g7e.2xlarge:
#        - stores the Hugging Face token in Secrets Manager (asks for it the
#          first time; the FLUX 3 weights are gated),
#        - uploads the serving code from ../FLUX3-Klein-sagemaker-g7e,
#        - stages the weights Hub -> S3 with a SageMaker Processing job (once),
#        - creates the SageMaker Model + EndpointConfig `FluxAiBoot-flux3-g7e`,
#        - adds it to the booth's managed configs, so an admin can make it
#          current and start it from the Endpoint tab like any other config.
#      The endpoint itself is NOT started here: the booth starts and stops it
#      (Endpoint tab or schedule), so no g7e instance is billed by a deploy.
#      With --endpoint-regions, the same config is also created in each extra
#      region (weights copied there from the home bucket), so the admin can run
#      the endpoint wherever g7e capacity is available.
#   6. sets PERMANENT passwords for the initial admin/visitor users so they can
#      sign in immediately (AdminCreateUser otherwise leaves them in
#      FORCE_CHANGE_PASSWORD)
#
# Re-runnable: every step is idempotent. Safe to run after any code change.
#
# Usage:
#   ./deploy.sh --admin-password <pw> --visitor-password <pw> [options]
#
# The initial admin and visitor passwords are REQUIRED and must be supplied
# explicitly (there are no built-in defaults, so no credentials live in this
# repo). Pass them as flags:
#
#   ./deploy.sh --admin-password 'S3cret!Admin' --visitor-password 'S3cret!Visit'
#
# or via environment variables:
#
#   ADMIN_PASSWORD='S3cret!Admin' VISITOR_PASSWORD='S3cret!Visit' ./deploy.sh
#
# On a re-deploy where the users already have their passwords, pass
# --skip-passwords instead (step 6 is skipped and nobody's password changes).
#
# Passwords must satisfy the Cognito password policy (min 8 chars, and at least
# one uppercase letter, one lowercase letter, and one special character).
#
# Every other value has a sensible default and can be overridden with the
# matching flag or environment variable, e.g.:
#   ./deploy.sh --region us-east-1 --sender-email me@example.com \
#       --admin-password ... --visitor-password ...
#
# Run `./deploy.sh --help` for the full flag list.
#
set -euo pipefail

# --- Configuration defaults (override via flag or environment) ---------------
REGION="${REGION:-us-east-1}"
STACK_NAME="${STACK_NAME:-AiPhotoBoothStack}"

# The shared SageMaker bucket the FLUX endpoint reads/writes. Referenced (not
# created) by the stack, so it MUST already exist in the TARGET account/region.
# Leave unset to auto-default to that account's standard SageMaker default
# bucket: sagemaker-<region>-<account-id> (resolved after flag parsing). Pointing
# at a bucket in a DIFFERENT account causes the CORS custom resource to fail with
# AccessDenied. Override with --existing-io-bucket if your endpoint uses another.
EXISTING_IO_BUCKET="${EXISTING_IO_BUCKET:-}"

ENDPOINT_NAME="${ENDPOINT_NAME:-flux2-klein-9b-g6e2}"
ENDPOINT_CONFIG_NAME="${ENDPOINT_CONFIG_NAME:-flux2-klein-9b-g6e2}"
ADMIN_USERNAME="${ADMIN_USERNAME:-admin}"
STANDARD_USERNAME="${STANDARD_USERNAME:-visitor}"
SCHEDULER_TIMEZONE="${SCHEDULER_TIMEZONE:-Asia/Jerusalem}"

# Permanent passwords set after deploy so the initial users can sign in.
# REQUIRED — no defaults — unless --skip-passwords. Supply via
# --admin-password / --visitor-password or the ADMIN_PASSWORD /
# VISITOR_PASSWORD environment variables.
ADMIN_PASSWORD="${ADMIN_PASSWORD:-}"
VISITOR_PASSWORD="${VISITOR_PASSWORD:-}"
SKIP_PASSWORDS="${SKIP_PASSWORDS:-false}"

# FLUX 3 Image Klein on ml.g7e.2xlarge (step 5). The config name must contain
# "flux3": the booth picks the FLUX 3 prompts and sampling from the name of the
# current config (ui/src/api/modelFamily.ts).
SKIP_FLUX3="${SKIP_FLUX3:-false}"
FLUX3_CONFIG_NAME="${FLUX3_CONFIG_NAME:-FluxAiBoot-flux3-g7e}"
FLUX3_INSTANCE_TYPE="${FLUX3_INSTANCE_TYPE:-ml.g7e.2xlarge}"
# AWS Ray Serve DLC for SageMaker (PyTorch 2.10 / CUDA 12.9, Blackwell-capable),
# pinned so serving/code/requirements.txt's guard pins stay true.
FLUX3_IMAGE_TAG="${FLUX3_IMAGE_TAG:-serve-ml-sagemaker-cuda-v1.5.7}"
FLUX3_DLC_ACCOUNT="${FLUX3_DLC_ACCOUNT:-763104351884}"
# NVIDIA driver 580 / CUDA 13.0 host AMI (required by the Ray DLC on GPU).
FLUX3_INFERENCE_AMI="${FLUX3_INFERENCE_AMI:-al2023-ami-sagemaker-inference-gpu-4-1}"
FLUX3_STAGING_INSTANCE_TYPE="${FLUX3_STAGING_INSTANCE_TYPE:-ml.c6i.4xlarge}"
FLUX3_HF_REPO="${FLUX3_HF_REPO:-black-forest-labs/flux-3-image-klein}"
# Hugging Face read token (license accepted on the repo). Asked for
# interactively when it is not already in Secrets Manager; --hf-token replaces
# the stored one.
HF_TOKEN="${HF_TOKEN:-}"

# Extra regions the admin may pick endpoints from (comma-separated, e.g.
# us-east-2). Each one uses its default SageMaker bucket
# (sagemaker-<region>-<account>, created if missing) for the booth's requests
# and results, and gets its own copy of the FLUX 3 endpoint config. Unset keeps
# the regions the deployed stack already has; "none" means the home region only.
ENDPOINT_REGIONS="${ENDPOINT_REGIONS:-}"

# --- Usage -------------------------------------------------------------------
usage() {
  cat <<'USAGE'
Usage: ./deploy.sh --admin-password <pw> --visitor-password <pw> [options]

Required (unless --skip-passwords):
  --admin-password <pw>      Permanent password for the initial admin user.
  --visitor-password <pw>    Permanent password for the initial visitor user.
                             (Both may also be supplied via the ADMIN_PASSWORD
                             and VISITOR_PASSWORD environment variables.)
  --skip-passwords           Re-deploy without setting the users' passwords.

Options (each has a default; flag overrides env var overrides default):
  --region <r>                       AWS region (default: us-east-1)
  --stack-name <name>                CloudFormation stack name
  --existing-io-bucket <bucket>      Shared SageMaker I/O bucket (referenced)
  --endpoint-name <name>             SageMaker endpoint name
  --endpoint-config-name <name>      SageMaker endpoint config name
  --admin-username <name>            Initial admin username (default: admin)
  --standard-username <name>         Initial visitor username (default: visitor)
  --scheduler-timezone <tz>          IANA timezone (default: Asia/Jerusalem)
  --endpoint-regions <r1,r2>         Extra regions the admin may pick endpoints
                                     from (e.g. us-east-2). Default: keep the
                                     deployed stack's; "none" = home region only.

FLUX 3 Image Klein (g7e) endpoint config:
  --hf-token <token>                 Hugging Face read token for the gated
                                     weights. Stored in Secrets Manager. Asked
                                     for interactively when not stored yet;
                                     pass it to replace the stored token.
                                     (Also: HF_TOKEN environment variable.)
  --flux3-config-name <name>         Endpoint config name; must contain "flux3"
                                     (default: FluxAiBoot-flux3-g7e)
  --flux3-instance-type <type>       Endpoint instance (default: ml.g7e.2xlarge)
  --skip-flux3                       Skip the FLUX 3 step entirely.
  -h, --help                         Show this help and exit

Passwords must satisfy the Cognito password policy: at least 8 characters with
at least one uppercase letter, one lowercase letter, and one special character.
USAGE
}

# --- Parse CLI flags ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --admin-password) ADMIN_PASSWORD="$2"; shift 2 ;;
    --visitor-password) VISITOR_PASSWORD="$2"; shift 2 ;;
    --skip-passwords) SKIP_PASSWORDS=true; shift ;;
    --region) REGION="$2"; shift 2 ;;
    --stack-name) STACK_NAME="$2"; shift 2 ;;
    --existing-io-bucket) EXISTING_IO_BUCKET="$2"; shift 2 ;;
    --endpoint-name) ENDPOINT_NAME="$2"; shift 2 ;;
    --endpoint-config-name) ENDPOINT_CONFIG_NAME="$2"; shift 2 ;;
    --admin-username) ADMIN_USERNAME="$2"; shift 2 ;;
    --standard-username) STANDARD_USERNAME="$2"; shift 2 ;;
    --scheduler-timezone) SCHEDULER_TIMEZONE="$2"; shift 2 ;;
    --endpoint-regions) ENDPOINT_REGIONS="$2"; shift 2 ;;
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    --flux3-config-name) FLUX3_CONFIG_NAME="$2"; shift 2 ;;
    --flux3-instance-type) FLUX3_INSTANCE_TYPE="$2"; shift 2 ;;
    --skip-flux3) SKIP_FLUX3=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; echo >&2; usage >&2; exit 2 ;;
  esac
done

# --- Validate required passwords ---------------------------------------------
if [[ "$SKIP_PASSWORDS" != true && ( -z "$ADMIN_PASSWORD" || -z "$VISITOR_PASSWORD" ) ]]; then
  echo "ERROR: --admin-password and --visitor-password are required (or set" >&2
  echo "       ADMIN_PASSWORD / VISITOR_PASSWORD in the environment)." >&2
  echo "       On a re-deploy that should leave passwords alone, pass --skip-passwords." >&2
  echo >&2
  usage >&2
  exit 2
fi

# --- Resolve the I/O bucket default to the TARGET account, if not supplied ---
# Avoids the cross-account trap: a hardcoded bucket from another account makes
# the CORS custom resource fail with AccessDenied. Defaults to the deploying
# account's standard SageMaker bucket (sagemaker-<region>-<account-id>).
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
if [[ -z "$EXISTING_IO_BUCKET" ]]; then
  if [[ -z "$ACCOUNT_ID" ]]; then
    echo "ERROR: could not resolve the AWS account id (aws sts get-caller-identity)." >&2
    echo "       Check your credentials, or pass --existing-io-bucket explicitly." >&2
    exit 2
  fi
  EXISTING_IO_BUCKET="sagemaker-${REGION}-${ACCOUNT_ID}"
  echo "==> Using I/O bucket (default for this account): $EXISTING_IO_BUCKET"
fi

# --- Extra endpoint regions --------------------------------------------------
# Unset: keep what the deployed stack has (output EndpointRegions, written as
# "home=bucket,extra=bucket"), so a plain re-deploy never drops a region.
if [[ -z "$ENDPOINT_REGIONS" ]]; then
  DEPLOYED_REGIONS="$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='EndpointRegions'].OutputValue" --output text 2>/dev/null || true)"
  if [[ -n "$DEPLOYED_REGIONS" && "$DEPLOYED_REGIONS" != "None" ]]; then
    ENDPOINT_REGIONS="$(printf '%s' "$DEPLOYED_REGIONS" | tr ',' '\n' | cut -d= -f1 | paste -sd, -)"
  fi
elif [[ "$ENDPOINT_REGIONS" == none ]]; then
  ENDPOINT_REGIONS=""
fi
EXTRA_REGION_LIST=""
for r in $(printf '%s' "$ENDPOINT_REGIONS" | tr ',' ' '); do
  if [[ ! "$r" =~ ^[a-z]{2}(-[a-z]+)+-[0-9]$ ]]; then
    echo "ERROR: --endpoint-regions: \"$r\" is not an AWS region name." >&2
    exit 2
  fi
  if [[ "$r" != "$REGION" && " $EXTRA_REGION_LIST " != *" $r "* ]]; then
    EXTRA_REGION_LIST="${EXTRA_REGION_LIST:+$EXTRA_REGION_LIST }$r"
  fi
done
EXTRA_REGIONS_CSV="$(printf '%s' "$EXTRA_REGION_LIST" | tr ' ' ',')"
if [[ -n "$EXTRA_REGION_LIST" ]]; then
  if [[ -z "$ACCOUNT_ID" ]]; then
    echo "ERROR: could not resolve the AWS account id, needed for the extra regions' buckets." >&2
    exit 2
  fi
  echo "==> Extra endpoint regions: $EXTRA_REGIONS_CSV"
  for r in $EXTRA_REGION_LIST; do
    b="sagemaker-$r-$ACCOUNT_ID"
    if aws s3api head-bucket --bucket "$b" --region "$r" >/dev/null 2>&1; then
      echo "    I/O bucket for $r: $b"
    else
      echo "    Creating the I/O bucket for $r: $b"
      if [[ "$r" == us-east-1 ]]; then
        aws s3api create-bucket --bucket "$b" --region "$r" >/dev/null
      else
        aws s3api create-bucket --bucket "$b" --region "$r" \
          --create-bucket-configuration LocationConstraint="$r" >/dev/null
      fi
      aws s3api put-public-access-block --bucket "$b" --region "$r" --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    fi
  done
fi

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="$SCRIPT_DIR/ui"
CDK_DIR="$SCRIPT_DIR/cdk"
FLUX3_DIR="${FLUX3_DIR:-$SCRIPT_DIR/../FLUX3-Klein-sagemaker-g7e}"

# Same names as cdk/lib/flux3-construct.ts and the standalone notebook.
FLUX3_PREFIX="flux3-klein-g7e"
FLUX3_SECRET_NAME="AmazonSageMaker-flux3-klein-hf-token"

# Scratch files (the HF token never goes on a command line); removed on exit.
WORK_DIR="$(mktemp -d)"
chmod 700 "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT

# --- FLUX 3 preflight: settle the Hugging Face token BEFORE the long build ---
FLUX3_SECRET_EXISTS=false
if [[ "$SKIP_FLUX3" != true ]]; then
  for f in serving/config.yaml serving/code/flux3_app.py serving/code/requirements.txt scripts/stage_weights.py; do
    if [[ ! -f "$FLUX3_DIR/$f" ]]; then
      echo "ERROR: $FLUX3_DIR/$f not found (the FLUX 3 serving code). Set FLUX3_DIR or pass --skip-flux3." >&2
      exit 2
    fi
  done
  if [[ "$FLUX3_CONFIG_NAME" != *[Ff][Ll][Uu][Xx]3* && "$FLUX3_CONFIG_NAME" != *[Ff][Ll][Uu][Xx]-3* ]]; then
    echo "ERROR: --flux3-config-name must contain \"flux3\" so the booth sends FLUX 3 prompts to it." >&2
    exit 2
  fi
  if aws secretsmanager describe-secret --secret-id "$FLUX3_SECRET_NAME" --region "$REGION" >/dev/null 2>&1; then
    FLUX3_SECRET_EXISTS=true
  fi
  if [[ -z "$HF_TOKEN" && "$FLUX3_SECRET_EXISTS" != true ]]; then
    if [[ -t 0 ]]; then
      echo "==> FLUX 3 Image Klein needs a Hugging Face read token (accept the license at"
      echo "    https://huggingface.co/$FLUX3_HF_REPO first). It is stored in Secrets Manager"
      echo "    as $FLUX3_SECRET_NAME and only used to copy the weights to S3."
      read -rsp "    Hugging Face token: " HF_TOKEN
      echo
    fi
    if [[ -z "$HF_TOKEN" ]]; then
      echo "ERROR: no Hugging Face token. Pass --hf-token, set HF_TOKEN, run interactively, or --skip-flux3." >&2
      exit 2
    fi
  fi
  if [[ -n "$HF_TOKEN" ]]; then
    # Prove the token can read the gated repo now, not 20 minutes into a deploy.
    printf 'Authorization: Bearer %s\n' "$HF_TOKEN" > "$WORK_DIR/hf-header"
    HF_STATUS="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -H "@$WORK_DIR/hf-header" \
      "https://huggingface.co/$FLUX3_HF_REPO/resolve/main/modular_model_index.json" || true)"
    rm -f "$WORK_DIR/hf-header"
    if [[ "$HF_STATUS" != "200" ]]; then
      echo "ERROR: the Hugging Face token cannot read $FLUX3_HF_REPO (HTTP $HF_STATUS)." >&2
      echo "       Accept the license on the model page with the token's account, or check the token." >&2
      exit 2
    fi
    echo "==> Hugging Face token can read $FLUX3_HF_REPO"
  else
    echo "==> Using the Hugging Face token already stored in Secrets Manager ($FLUX3_SECRET_NAME)"
  fi
fi

echo "==> AI Photo Booth deploy (region=$REGION, stack=$STACK_NAME)"

# --- 1. Build the SPA --------------------------------------------------------
echo "==> [1/6] Building UI (ui/)"
( cd "$UI_DIR" && npm run build )

# --- 2. Build the CDK app ----------------------------------------------------
echo "==> [2/6] Building CDK (cdk/)"
( cd "$CDK_DIR" && npm run build )

# --- 3. Deploy ---------------------------------------------------------------
echo "==> [3/6] Deploying $STACK_NAME"
(
  cd "$CDK_DIR"
  CDK_DEFAULT_REGION="$REGION" npx cdk deploy \
    --require-approval never \
    --context existingIoBucketName="$EXISTING_IO_BUCKET" \
    --context endpointRegions="$EXTRA_REGIONS_CSV" \
    --parameters EndpointName="$ENDPOINT_NAME" \
    --parameters EndpointConfigName="$ENDPOINT_CONFIG_NAME" \
    --parameters InitialAdminUsername="$ADMIN_USERNAME" \
    --parameters InitialStandardUsername="$STANDARD_USERNAME" \
    --parameters SchedulerTimezone="$SCHEDULER_TIMEZONE"
)

# --- 4. Read outputs + regenerate ui/.env.local ------------------------------
echo "==> [4/6] Reading stack outputs and regenerating ui/.env.local"

get_output() {
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" \
    --output text
}

USER_POOL_ID="$(get_output UserPoolId)"
USER_POOL_CLIENT_ID="$(get_output UserPoolClientId)"
IDENTITY_POOL_ID="$(get_output IdentityPoolId)"
IO_BUCKET="$(get_output IoBucketName)"
SCHEDULE_TABLE="$(get_output ScheduleTableName)"
ENDPOINT_NAME_OUT="$(get_output EndpointNameOutput)"
ENDPOINT_CONFIG_OUT="$(get_output EndpointConfigNameOutput)"
INVOKE_FUNCTION_URL="$(get_output InvokeFunctionUrl)"
DISTRIBUTION_DOMAIN="$(get_output DistributionDomainName)"
SHARE_BUCKET="$(get_output ShareBucketName)"
SHARE_SIGNER_URL="$(get_output ShareSignerUrl)"
FLUX3_ROLE_ARN="$(get_output Flux3SageMakerRoleArn)"

cat > "$UI_DIR/.env.local" <<EOF
# Local-dev runtime config for \`npm run dev\`.
# AUTO-GENERATED by deploy.sh from the deployed $STACK_NAME outputs.
# Vite injects these as import.meta.env.*, which src/config.ts reads as a
# fallback when window.__BOOTH_CONFIG__ is absent (i.e. outside CloudFront).
VITE_AWS_REGION=$REGION
VITE_COGNITO_USER_POOL_ID=$USER_POOL_ID
VITE_COGNITO_USER_POOL_CLIENT_ID=$USER_POOL_CLIENT_ID
VITE_COGNITO_IDENTITY_POOL_ID=$IDENTITY_POOL_ID
VITE_ENDPOINT_NAME=$ENDPOINT_NAME_OUT
VITE_ENDPOINT_CONFIG_NAME=$ENDPOINT_CONFIG_OUT
VITE_IO_BUCKET=$IO_BUCKET
VITE_SCHEDULE_TABLE=$SCHEDULE_TABLE
VITE_SHARE_BUCKET=$SHARE_BUCKET
VITE_SHARE_SIGNER_URL=$SHARE_SIGNER_URL
VITE_TIMEZONE=$SCHEDULER_TIMEZONE
# IAM-authenticated Invoke_Proxy Function URL (SageMaker runtime has no CORS, so
# the browser cannot call InvokeEndpointAsync directly).
VITE_INVOKE_FUNCTION_URL=$INVOKE_FUNCTION_URL
EOF

echo "    Wrote $UI_DIR/.env.local"

# --- 5. FLUX 3 Image Klein endpoint config (g7e) -----------------------------
if [[ "$SKIP_FLUX3" == true ]]; then
  echo "==> [5/6] Skipping FLUX 3 (--skip-flux3)"
else
  echo "==> [5/6] FLUX 3 Image Klein endpoint config '$FLUX3_CONFIG_NAME' ($FLUX3_INSTANCE_TYPE)"
  MODEL_PREFIX="$FLUX3_PREFIX/model"
  MODEL_URI="s3://$IO_BUCKET/$MODEL_PREFIX/"
  IMAGE_URI="$FLUX3_DLC_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ray:$FLUX3_IMAGE_TAG"
  if [[ -z "$FLUX3_ROLE_ARN" || "$FLUX3_ROLE_ARN" == "None" ]]; then
    echo "ERROR: stack output Flux3SageMakerRoleArn is missing." >&2
    exit 1
  fi

  # 5a. Hugging Face token -> Secrets Manager (file:// keeps it off the command line).
  if [[ -n "$HF_TOKEN" ]]; then
    printf '%s' "$HF_TOKEN" > "$WORK_DIR/hf-token"
    if [[ "$FLUX3_SECRET_EXISTS" == true ]]; then
      aws secretsmanager put-secret-value --secret-id "$FLUX3_SECRET_NAME" \
        --secret-string "file://$WORK_DIR/hf-token" --region "$REGION" >/dev/null
      echo "    Updated the Hugging Face token in Secrets Manager ($FLUX3_SECRET_NAME)"
    else
      aws secretsmanager create-secret --name "$FLUX3_SECRET_NAME" \
        --description "Hugging Face read token for staging FLUX 3 Image Klein weights (AI Photo Booth)" \
        --secret-string "file://$WORK_DIR/hf-token" \
        --tags Key=SageMaker,Value=true Key=AiPhoto,Value=true --region "$REGION" >/dev/null
      echo "    Stored the Hugging Face token in Secrets Manager ($FLUX3_SECRET_NAME)"
    fi
    rm -f "$WORK_DIR/hf-token"
  fi

  # 5b. Serving code next to the weights: the Ray Serve DLC runs
  #     `serve run /opt/ml/model/config.yaml` after installing code/requirements.txt.
  flux3_upload_serving() {  # <region> <bucket>
    aws s3 cp --only-show-errors "$FLUX3_DIR/serving/config.yaml" "s3://$2/$MODEL_PREFIX/config.yaml" --region "$1"
    for f in flux3_app.py requirements.txt; do
      aws s3 cp --only-show-errors "$FLUX3_DIR/serving/code/$f" "s3://$2/$MODEL_PREFIX/code/$f" --region "$1"
    done
    echo "    Uploaded the serving code to s3://$2/$MODEL_PREFIX/"
  }
  flux3_upload_serving "$REGION" "$IO_BUCKET"
  aws s3 cp --only-show-errors "$FLUX3_DIR/scripts/stage_weights.py" \
    "s3://$IO_BUCKET/$FLUX3_PREFIX/staging/code/stage_weights.py" --region "$REGION"

  # 5c. Weights (~39 GB), staged once by a CPU Processing job inside AWS. The
  #     manifest is written last, so its presence means staging completed.
  if aws s3api head-object --bucket "$IO_BUCKET" --key "$MODEL_PREFIX/weights/_flux3_stage.json" \
      --region "$REGION" >/dev/null 2>&1; then
    echo "    Weights already staged under ${MODEL_URI}weights/"
  else
    STAGE_JOB="flux3-stage-$(date -u +%Y%m%d-%H%M%S)"
    cat > "$WORK_DIR/stage-job.json" <<EOF
{
  "ProcessingJobName": "$STAGE_JOB",
  "RoleArn": "$FLUX3_ROLE_ARN",
  "AppSpecification": {
    "ImageUri": "$IMAGE_URI",
    "ContainerEntrypoint": ["python", "/opt/ml/processing/input/code/stage_weights.py"],
    "ContainerArguments": ["--repo", "$FLUX3_HF_REPO", "--bucket", "$IO_BUCKET",
                           "--prefix", "$MODEL_PREFIX/weights", "--region", "$REGION",
                           "--scratch", "/opt/ml/processing/scratch",
                           "--hf-token-secret", "$FLUX3_SECRET_NAME"]
  },
  "ProcessingInputs": [{
    "InputName": "code",
    "S3Input": {"S3Uri": "s3://$IO_BUCKET/$FLUX3_PREFIX/staging/code/stage_weights.py",
                "LocalPath": "/opt/ml/processing/input/code", "S3DataType": "S3Prefix", "S3InputMode": "File"}
  }],
  "ProcessingResources": {"ClusterConfig": {"InstanceCount": 1, "InstanceType": "$FLUX3_STAGING_INSTANCE_TYPE", "VolumeSizeInGB": 120}},
  "StoppingCondition": {"MaxRuntimeInSeconds": 7200},
  "Environment": {"AWS_REGION": "$REGION", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"},
  "Tags": [{"Key": "AiPhoto", "Value": "true"}]
}
EOF
    # A just-created role can take a few seconds to be usable by SageMaker.
    for attempt in 1 2 3 4 5; do
      if aws sagemaker create-processing-job --cli-input-json "file://$WORK_DIR/stage-job.json" \
          --region "$REGION" >/dev/null 2>"$WORK_DIR/err"; then
        break
      fi
      if [[ $attempt -eq 5 ]] || ! grep -qi "role" "$WORK_DIR/err"; then
        cat "$WORK_DIR/err" >&2
        exit 1
      fi
      sleep $((attempt * 15))
    done
    echo "    Staging the weights from Hugging Face: Processing job $STAGE_JOB (about 10 minutes)"
    aws sagemaker wait processing-job-completed-or-stopped --processing-job-name "$STAGE_JOB" --region "$REGION"
    STAGE_STATUS="$(aws sagemaker describe-processing-job --processing-job-name "$STAGE_JOB" \
      --region "$REGION" --query ProcessingJobStatus --output text)"
    if [[ "$STAGE_STATUS" != "Completed" ]] || ! aws s3api head-object --bucket "$IO_BUCKET" \
        --key "$MODEL_PREFIX/weights/_flux3_stage.json" --region "$REGION" >/dev/null 2>&1; then
      echo "ERROR: weight staging $STAGE_STATUS. Logs: aws logs tail /aws/sagemaker/ProcessingJobs --log-stream-name-prefix $STAGE_JOB" >&2
      aws sagemaker describe-processing-job --processing-job-name "$STAGE_JOB" --region "$REGION" \
        --query '[FailureReason,ExitMessage]' --output text >&2 || true
      exit 1
    fi
    echo "    Weights staged under ${MODEL_URI}weights/"
  fi

  # 5d. Model + EndpointConfig in <region>, reading model data from and
  #     writing results to <bucket>. The booth creates the endpoint FROM this
  #     config (same name), so leave them alone while an endpoint of that name
  #     exists in that region.
  flux3_model_config() {  # <region> <bucket>
    local r="$1" b="$2"
    local image="$FLUX3_DLC_ACCOUNT.dkr.ecr.$1.amazonaws.com/ray:$FLUX3_IMAGE_TAG"
    local model_uri="s3://$2/$MODEL_PREFIX/"
    FLUX3_ENDPOINT_STATUS="$(aws sagemaker describe-endpoint --endpoint-name "$FLUX3_CONFIG_NAME" \
      --region "$r" --query EndpointStatus --output text 2>/dev/null || echo NONE)"
    if [[ "$FLUX3_ENDPOINT_STATUS" != "NONE" ]]; then
      echo "    Endpoint '$FLUX3_CONFIG_NAME' exists in $r ($FLUX3_ENDPOINT_STATUS): its model and config are left as they are."
      echo "    Stop it from the booth and re-run deploy.sh to apply image or environment changes."
      echo "    (Serving code changes in S3 apply the next time the booth starts it.)"
    else
      aws sagemaker delete-endpoint-config --endpoint-config-name "$FLUX3_CONFIG_NAME" --region "$r" >/dev/null 2>&1 || true
      aws sagemaker delete-model --model-name "$FLUX3_CONFIG_NAME" --region "$r" >/dev/null 2>&1 || true
      cat > "$WORK_DIR/model.json" <<EOF
{
  "ModelName": "$FLUX3_CONFIG_NAME",
  "ExecutionRoleArn": "$FLUX3_ROLE_ARN",
  "PrimaryContainer": {
    "Image": "$image",
    "ModelDataSource": {"S3DataSource": {"S3Uri": "$model_uri", "S3DataType": "S3Prefix", "CompressionType": "None"}},
    "Environment": {
      "FLUX3_WEIGHTS_DIR": "/opt/ml/model/weights",
      "RAYSERVE_BACKEND_TIMEOUT": "3600",
      "HF_HUB_OFFLINE": "1",
      "TRANSFORMERS_OFFLINE": "1",
      "HF_HOME": "/tmp/hf",
      "TRITON_CACHE_DIR": "/tmp/triton",
      "RAY_USAGE_STATS_ENABLED": "0",
      "PIP_DISABLE_PIP_VERSION_CHECK": "1"
    }
  },
  "Tags": [{"Key": "AiPhoto", "Value": "true"}]
}
EOF
      for attempt in 1 2 3 4 5; do
        if aws sagemaker create-model --cli-input-json "file://$WORK_DIR/model.json" \
            --region "$r" >/dev/null 2>"$WORK_DIR/err"; then
          break
        fi
        if [[ $attempt -eq 5 ]] || ! grep -qi "role" "$WORK_DIR/err"; then
          cat "$WORK_DIR/err" >&2
          exit 1
        fi
        sleep $((attempt * 15))
      done
      # Same async I/O prefixes as the FLUX.2 endpoint: the booth uploads requests
      # to flux2-klein-inputs/ and polls flux2-klein-outputs/ + flux2-klein-failures/.
      cat > "$WORK_DIR/config.json" <<EOF
{
  "EndpointConfigName": "$FLUX3_CONFIG_NAME",
  "ProductionVariants": [{
    "VariantName": "AllTraffic",
    "ModelName": "$FLUX3_CONFIG_NAME",
    "InstanceType": "$FLUX3_INSTANCE_TYPE",
    "InitialInstanceCount": 1,
    "ModelDataDownloadTimeoutInSeconds": 3600,
    "ContainerStartupHealthCheckTimeoutInSeconds": 3600,
    "InferenceAmiVersion": "$FLUX3_INFERENCE_AMI"
  }],
  "AsyncInferenceConfig": {
    "OutputConfig": {"S3OutputPath": "s3://$b/flux2-klein-outputs/",
                     "S3FailurePath": "s3://$b/flux2-klein-failures/"},
    "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 1}
  },
  "Tags": [{"Key": "AiPhoto", "Value": "true"}]
}
EOF
      aws sagemaker create-endpoint-config --cli-input-json "file://$WORK_DIR/config.json" --region "$r" >/dev/null
      echo "    Created SageMaker model + endpoint config '$FLUX3_CONFIG_NAME' in $r"
    fi
  }
  flux3_model_config "$REGION" "$IO_BUCKET"

  # 5e. Offer it in the booth's Endpoint tab (managed configs). Never touches
  #     the current-config pointer: the admin decides when to switch. The
  #     table lives in the home region; a config in another region is keyed
  #     CONFIG#<region>#<name> (ui/src/api/schedule.ts managedConfigSk).
  flux3_register() {  # <endpoint region>
    local r="$1" sk="CONFIG#$FLUX3_CONFIG_NAME" label="'$FLUX3_CONFIG_NAME'"
    if [[ "$r" != "$REGION" ]]; then
      sk="CONFIG#$r#$FLUX3_CONFIG_NAME"
      label="'$FLUX3_CONFIG_NAME' ($r)"
    fi
    local now_iso
    now_iso="$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"
    if aws dynamodb put-item --table-name "$SCHEDULE_TABLE" --region "$REGION" \
        --condition-expression "attribute_not_exists(pk)" \
        --item "{\"pk\":{\"S\":\"CONFIG#MANAGED\"},\"sk\":{\"S\":\"$sk\"},\"configName\":{\"S\":\"$FLUX3_CONFIG_NAME\"},\"region\":{\"S\":\"$r\"},\"addedAt\":{\"S\":\"$now_iso\"}}" \
        >/dev/null 2>"$WORK_DIR/err"; then
      echo "    Added $label to the booth's managed endpoint configs"
    elif grep -q ConditionalCheckFailed "$WORK_DIR/err"; then
      echo "    $label is already one of the booth's managed endpoint configs"
    else
      cat "$WORK_DIR/err" >&2
      exit 1
    fi
  }
  flux3_register "$REGION"

  # 5f. The same endpoint config in each extra region. The weights are copied
  #     from the home bucket server-side (S3 to S3) once; the manifest is copied
  #     last, so its presence means the copy completed.
  for XR in $EXTRA_REGION_LIST; do
    XB="sagemaker-$XR-$ACCOUNT_ID"
    echo "    --- $XR (bucket $XB)"
    if aws s3api head-object --bucket "$XB" --key "$MODEL_PREFIX/weights/_flux3_stage.json" \
        --region "$XR" >/dev/null 2>&1; then
      echo "    Weights already staged under s3://$XB/$MODEL_PREFIX/weights/"
    else
      echo "    Copying the staged weights to s3://$XB/$MODEL_PREFIX/weights/ (about 39 GB, S3 to S3, 10-20 minutes)"
      aws s3 sync --only-show-errors "s3://$IO_BUCKET/$MODEL_PREFIX/weights/" "s3://$XB/$MODEL_PREFIX/weights/" \
        --source-region "$REGION" --region "$XR" --exclude "_flux3_stage.json"
      aws s3 cp --only-show-errors "s3://$IO_BUCKET/$MODEL_PREFIX/weights/_flux3_stage.json" \
        "s3://$XB/$MODEL_PREFIX/weights/_flux3_stage.json" --source-region "$REGION" --region "$XR"
    fi
    flux3_upload_serving "$XR" "$XB"
    flux3_model_config "$XR" "$XB"
    flux3_register "$XR"
  done
fi

# --- 6. Set permanent passwords for the initial users ------------------------
if [[ "$SKIP_PASSWORDS" == true ]]; then
  echo "==> [6/6] Skipping passwords (--skip-passwords); existing users keep theirs"
else
  echo "==> [6/6] Setting permanent passwords for initial users"

  set_password() {
    local username="$1" password="$2"
    if aws cognito-idp admin-set-user-password \
        --user-pool-id "$USER_POOL_ID" \
        --username "$username" \
        --password "$password" \
        --permanent \
        --region "$REGION" >/dev/null 2>&1; then
      echo "    Set permanent password for '$username'"
    else
      echo "    WARN: could not set password for '$username' (user may not exist yet)"
    fi
  }

  set_password "$ADMIN_USERNAME" "$ADMIN_PASSWORD"
  set_password "$STANDARD_USERNAME" "$VISITOR_PASSWORD"
fi

# --- Done --------------------------------------------------------------------
echo ""
echo "==> Deploy complete."
echo "    App URL:            https://$DISTRIBUTION_DOMAIN"
echo "    Invoke Function URL: $INVOKE_FUNCTION_URL"
echo "    Admin user:         $ADMIN_USERNAME"
echo "    Visitor user:       $STANDARD_USERNAME"
if [[ "$SKIP_FLUX3" != true ]]; then
  echo "    FLUX 3 endpoint:    '$FLUX3_CONFIG_NAME' ($FLUX3_INSTANCE_TYPE). In the booth: Admin -> Endpoint ->"
  echo "                        Make current, then Start (about 5 minutes; g7e capacity can be limited)."
  if [[ -n "$EXTRA_REGION_LIST" ]]; then
    echo "                        Also available in: $EXTRA_REGIONS_CSV (pick the row with that region)."
  fi
fi
echo ""
echo "    Run locally:        ( cd ui && npm run dev )  ->  http://localhost:5173"
