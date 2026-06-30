#!/bin/bash
# Deploy SAM3D Studio: CDK (Cognito + API + Lambdas + CloudFront), then set the
# two users' passwords OUT-OF-BAND (never in CloudFormation), then build & ship
# the React UI.
#
# Usage:
#   ./deploy.sh
# Optional env:
#   AWS_REGION (default: your CLI default)
#   ENDPOINT_NAME / ENDPOINT_CONFIG_NAME (default: sam3d-objects-g6e)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-$(aws configure get region || echo us-east-1)}"
ENDPOINT_NAME="${ENDPOINT_NAME:-sam3d-objects-g6e}"
ENDPOINT_CONFIG_NAME="${ENDPOINT_CONFIG_NAME:-sam3d-objects-g6e}"
export CDK_DEFAULT_REGION="$REGION"
export CDK_DEFAULT_ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"

echo "Account: $CDK_DEFAULT_ACCOUNT   Region: $REGION"

# ---------------------------------------------------------------- 1) CDK deploy
cd "$HERE/cdk"
[ -d node_modules ] || npm install
OUT="$HERE/cdk/outputs.json"
npx cdk deploy --require-approval never \
  -c endpointName="$ENDPOINT_NAME" \
  -c endpointConfigName="$ENDPOINT_CONFIG_NAME" \
  --outputs-file "$OUT"

get() { node -e "const o=require('$OUT');const s=Object.values(o)[0];process.stdout.write(s['$1']||'')"; }
USER_POOL_ID="$(get UserPoolId)"
CLIENT_ID="$(get UserPoolClientId)"
API_URL="$(get ApiUrl)"
DIST_DOMAIN="$(get DistributionDomain)"
DIST_ID="$(get DistributionId)"
SPA_BUCKET="$(get SpaBucketName)"

echo ""
echo "UserPool: $USER_POOL_ID   Client: $CLIENT_ID"
echo "API:      $API_URL"
echo "SPA:      $DIST_DOMAIN"

# ----------------------------------------------- 2) set passwords out-of-band
set_pw() {
  local user="$1" pw1 pw2
  while true; do
    read -r -s -p "Password for '$user' (>=12 chars, upper/lower/digit/symbol): " pw1; echo
    read -r -s -p "Confirm password for '$user': " pw2; echo
    if [ "$pw1" != "$pw2" ]; then echo "  passwords do not match, try again"; continue; fi
    if [ "${#pw1}" -lt 12 ]; then echo "  too short, try again"; continue; fi
    break
  done
  aws cognito-idp admin-set-user-password \
    --user-pool-id "$USER_POOL_ID" --username "$user" \
    --password "$pw1" --permanent --region "$REGION"
  echo "  set password for $user"
}
echo ""
echo "Set passwords for the provisioned users (input hidden; not stored anywhere):"
set_pw admin
set_pw visitor

# ------------------------------------------------------------- 3) build + ship UI
if [ -d "$HERE/ui" ]; then
  cd "$HERE/ui"
  [ -d node_modules ] || npm install
  # Runtime config consumed by the SPA (written to public/ before build).
  mkdir -p public
  cat > public/config.js <<EOF
window.__SAM3D_CONFIG__ = {
  region: "$REGION",
  userPoolId: "$USER_POOL_ID",
  userPoolClientId: "$CLIENT_ID",
  apiUrl: "$API_URL",
};
EOF
  npm run build
  aws s3 sync dist/ "s3://$SPA_BUCKET/" --delete --region "$REGION"
  # config.js is in dist after build (copied from public/); ensure it's fresh:
  aws s3 cp public/config.js "s3://$SPA_BUCKET/config.js" --region "$REGION"
  aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" >/dev/null
  echo ""
  echo "UI deployed."
else
  echo ""
  echo "NOTE: ui/ not present yet — infra is deployed. Add the React app under studio/ui/ and re-run."
fi

echo ""
echo "================================================================"
echo "Open: $DIST_DOMAIN"
echo "Sign in as 'admin' (can start/stop the endpoint) or 'visitor'."
echo "================================================================"
