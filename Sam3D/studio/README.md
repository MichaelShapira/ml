# SAM3D Studio — serverless React app for SAM 3D Objects

A modern, fully-authenticated web app that does what the notebook/Gradio flow does — **click an object → WebGPU mask → 3D Gaussian Splat → orbit it** — on **serverless AWS**, with **admin-controlled** start/stop of the GPU endpoint.

This directory is the **infrastructure + backend** (CDK) **and** the **React UI** (`ui/`), built + deployed by `deploy.sh`.

## Architecture (confirmed)

```
Browser (React SPA, WebGPU SAM, splat viewer)
  │  static assets (HTTPS)            │ JWT (Cognito)        │ presigned PUT/GET
  ▼                                   ▼                      ▼
CloudFront ── OAC ──► S3 (SPA, private)   API Gateway (REST) ── Cognito authorizer
                                              │  Lambda (least-priv roles)
                                              ├─ userApi:  /upload-url, /generate, /result, GET /endpoint
                                              └─ adminApi: POST /endpoint/start, /endpoint/stop   (admin only)
                                                     │ InvokeEndpointAsync / Create|Delete|DescribeEndpoint
                                                     ▼
                                   S3 I/O bucket ◄──► SageMaker Async Endpoint (g6e)
                                   (image, mask, .ply)
```

- **No public access without auth.** Self sign-up is **disabled** — only the two CDK-provisioned users exist. Every API route requires a valid Cognito JWT. The only thing served publicly is the static SPA shell (HTML/JS), which shows a **login screen** until authenticated and carries no data or AWS credentials.
- **Browser never holds AWS credentials.** It holds a Cognito JWT only. Large blobs (the base64 image/mask and the `.ply`, which can be tens of MB) move **browser ↔ S3 directly via short-lived presigned URLs**, never through API Gateway (10 MB / 29 s limits). The browser polls `GET /result` — the same pattern that fixed the tunnel 504.
- **Roles via a user attribute.** The User Pool defines a custom attribute **`custom:role`** (`admin` | `visitor`). It rides in the JWT; the `adminApi` Lambda authorizes start/stop by checking `custom:role === "admin"` and returns 403 otherwise.
- **Admin start/stop = real cost control.** "Stop" **deletes the endpoint** (zero GPU billing); "Start" **recreates it** from the existing endpoint-config. Both are admin-only. Visitors can read status and generate (a clear message appears if the endpoint is stopped).
- **Least privilege IAM.** Each Lambda has its own role scoped to exact ARNs/prefixes:
  - `userApi`: `s3:PutObject` on `…/sam3d-inputs/*`, `s3:GetObject`/`ListBucket` on `…/sam3d-outputs/*` + `…/sam3d-failures/*`, `sagemaker:InvokeEndpointAsync` + `DescribeEndpoint` on the one endpoint ARN.
  - `adminApi`: `sagemaker:CreateEndpoint`/`DeleteEndpoint`/`DescribeEndpoint` on the endpoint ARN + `DescribeEndpointConfig` on the config ARN.
  - Nothing uses `*` resources except `sagemaker:ListEndpoints`-free status reads (scoped) and `s3` is prefix-scoped.

## Endpoints (all require Cognito JWT)

| Method/route | Who | Action |
|---|---|---|
| `POST /upload-url` | any user | returns `{ jobId, uploadUrl, inputKey }` (presigned PUT for the request JSON) |
| `POST /generate` | any user | `InvokeEndpointAsync(InputLocation=inputKey)` → `{ outputKey, failureKey }` |
| `GET /result?outputKey=&failureKey=` | any user | `{ status: pending\|done\|error, downloadUrl?, error? }` (presigned GET for the `.ply`) |
| `GET /endpoint` | any user | `{ status }` (DescribeEndpoint; `Stopped` if it doesn't exist) |
| `POST /endpoint/start` | **admin** | recreate the endpoint from its config |
| `POST /endpoint/stop` | **admin** | delete the endpoint |

## Deploy

```bash
cd studio
./deploy.sh
```
`deploy.sh`:
1. `cdk deploy` (creates Cognito pool + `custom:role`, the **admin** and **visitor** users, API, Lambdas, S3 SPA bucket, CloudFront, and adds CORS to the I/O bucket).
2. **Prompts you for the admin and visitor passwords** (hidden input) and sets them with `admin-set-user-password --permanent`. Passwords are **never** placed in CDK/CloudFormation.
3. Builds `ui/`, writes the runtime `config.js` from stack outputs, syncs to the SPA bucket, and invalidates CloudFront.

### Config / context (override via `cdk.json` context or `-c`)
- `endpointName` (default `sam3d-objects-g6e`)
- `endpointConfigName` (default `sam3d-objects-g6e`)
- `ioBucketName` (default `sagemaker-<region>-<account>`)
- `adminEmail`, `visitorEmail` (for the two users; usernames are `admin` / `visitor`)

## Prerequisites
- The SAM 3D Objects async endpoint + its endpoint-config already exist (you built them via `../sam3d-sagemaker.ipynb`). This stack references them by name; it does **not** create the GPU endpoint.
- Node 20+, AWS CDK v2, and AWS creds for the target account (`cdk bootstrap` once per account/region).

## The React UI (`ui/`)

Vite + React + TypeScript + Tailwind SPA:
- **`src/auth.ts`** — Cognito sign-in (amazon-cognito-identity-js); uses the **ID token** (so `custom:role` reaches the API) and exposes `isAdmin`.
- **`src/sam.ts`** — WebGPU SAM (SlimSAM via transformers.js): encode once per image, decode per click; foreground/background refine.
- **`src/viewer.ts`** — interactive Gaussian-splat viewer (`@mkkellogg/gaussian-splats-3d`); drag to rotate, scroll to zoom.
- **`src/api.ts`** — typed client: `upload-url` → presigned PUT → `generate` → poll `result` → presigned GET. Polling keeps each request short.
- **`src/components/`** — `Login`, `Studio` (canvas click-to-mask + viewer), `AdminBar` (live endpoint status + admin-only Start/Stop).
- Config comes from `config.js` (written by `deploy.sh`) or `.env.local` for `npm run dev`.

Local dev: `cd ui && cp .env.local.example .env.local` (fill from a deployed stack), `npm install`, `npm run dev` → http://localhost:5173 (WebGPU needs a secure context; localhost qualifies).

## Security notes
- Self sign-up disabled; only provisioned users. MFA can be enabled in `cognito` props if desired.
- Presigned URLs are short-lived (default 5 min) and scoped to a single key.
- CloudFront uses OAC; the SPA bucket has Block Public Access on and no bucket policy beyond the OAC read.
- The I/O bucket CORS allows only the CloudFront origin for `PUT/GET/HEAD`.
- The `adminApi` Lambda re-checks the `custom:role` claim server-side — the route is not admin-gated by API Gateway alone.
