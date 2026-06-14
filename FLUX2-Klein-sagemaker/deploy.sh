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
#   5. sets PERMANENT passwords for the initial admin/visitor users so they can
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
# REQUIRED — no defaults. Supply via --admin-password / --visitor-password or
# the ADMIN_PASSWORD / VISITOR_PASSWORD environment variables.
ADMIN_PASSWORD="${ADMIN_PASSWORD:-}"
VISITOR_PASSWORD="${VISITOR_PASSWORD:-}"

# --- Usage -------------------------------------------------------------------
usage() {
  cat <<'USAGE'
Usage: ./deploy.sh --admin-password <pw> --visitor-password <pw> [options]

Required:
  --admin-password <pw>      Permanent password for the initial admin user.
  --visitor-password <pw>    Permanent password for the initial visitor user.
                             (Both may also be supplied via the ADMIN_PASSWORD
                             and VISITOR_PASSWORD environment variables.)

Options (each has a default; flag overrides env var overrides default):
  --region <r>                       AWS region (default: us-east-1)
  --stack-name <name>                CloudFormation stack name
  --existing-io-bucket <bucket>      Shared SageMaker I/O bucket (referenced)
  --endpoint-name <name>             SageMaker endpoint name
  --endpoint-config-name <name>      SageMaker endpoint config name
  --admin-username <name>            Initial admin username (default: admin)
  --standard-username <name>         Initial visitor username (default: visitor)
  --scheduler-timezone <tz>          IANA timezone (default: Asia/Jerusalem)
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
    --region) REGION="$2"; shift 2 ;;
    --stack-name) STACK_NAME="$2"; shift 2 ;;
    --existing-io-bucket) EXISTING_IO_BUCKET="$2"; shift 2 ;;
    --endpoint-name) ENDPOINT_NAME="$2"; shift 2 ;;
    --endpoint-config-name) ENDPOINT_CONFIG_NAME="$2"; shift 2 ;;
    --admin-username) ADMIN_USERNAME="$2"; shift 2 ;;
    --standard-username) STANDARD_USERNAME="$2"; shift 2 ;;
    --scheduler-timezone) SCHEDULER_TIMEZONE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; echo >&2; usage >&2; exit 2 ;;
  esac
done

# --- Validate required passwords ---------------------------------------------
if [[ -z "$ADMIN_PASSWORD" || -z "$VISITOR_PASSWORD" ]]; then
  echo "ERROR: --admin-password and --visitor-password are required (or set" >&2
  echo "       ADMIN_PASSWORD / VISITOR_PASSWORD in the environment)." >&2
  echo >&2
  usage >&2
  exit 2
fi

# --- Resolve the I/O bucket default to the TARGET account, if not supplied ---
# Avoids the cross-account trap: a hardcoded bucket from another account makes
# the CORS custom resource fail with AccessDenied. Defaults to the deploying
# account's standard SageMaker bucket (sagemaker-<region>-<account-id>).
if [[ -z "$EXISTING_IO_BUCKET" ]]; then
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"
  if [[ -z "$ACCOUNT_ID" ]]; then
    echo "ERROR: could not resolve the AWS account id (aws sts get-caller-identity)." >&2
    echo "       Check your credentials, or pass --existing-io-bucket explicitly." >&2
    exit 2
  fi
  EXISTING_IO_BUCKET="sagemaker-${REGION}-${ACCOUNT_ID}"
  echo "==> Using I/O bucket (default for this account): $EXISTING_IO_BUCKET"
fi

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="$SCRIPT_DIR/ui"
CDK_DIR="$SCRIPT_DIR/cdk"

echo "==> AI Photo Booth deploy (region=$REGION, stack=$STACK_NAME)"

# --- 1. Build the SPA --------------------------------------------------------
echo "==> [1/5] Building UI (ui/)"
( cd "$UI_DIR" && npm run build )

# --- 2. Build the CDK app ----------------------------------------------------
echo "==> [2/5] Building CDK (cdk/)"
( cd "$CDK_DIR" && npm run build )

# --- 3. Deploy ---------------------------------------------------------------
echo "==> [3/5] Deploying $STACK_NAME"
(
  cd "$CDK_DIR"
  CDK_DEFAULT_REGION="$REGION" npx cdk deploy \
    --require-approval never \
    --context existingIoBucketName="$EXISTING_IO_BUCKET" \
    --parameters EndpointName="$ENDPOINT_NAME" \
    --parameters EndpointConfigName="$ENDPOINT_CONFIG_NAME" \
    --parameters InitialAdminUsername="$ADMIN_USERNAME" \
    --parameters InitialStandardUsername="$STANDARD_USERNAME" \
    --parameters SchedulerTimezone="$SCHEDULER_TIMEZONE"
)

# --- 4. Read outputs + regenerate ui/.env.local ------------------------------
echo "==> [4/5] Reading stack outputs and regenerating ui/.env.local"

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

# --- 5. Set permanent passwords for the initial users ------------------------
echo "==> [5/5] Setting permanent passwords for initial users"

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

# --- Done --------------------------------------------------------------------
echo ""
echo "==> Deploy complete."
echo "    App URL:            https://$DISTRIBUTION_DOMAIN"
echo "    Invoke Function URL: $INVOKE_FUNCTION_URL"
echo "    Admin user:         $ADMIN_USERNAME"
echo "    Visitor user:       $STANDARD_USERNAME"
echo ""
echo "    Run locally:        ( cd ui && npm run dev )  ->  http://localhost:5173"
