# AI Photo Booth — Development Notes & Skill Doc

Hard-won knowledge from building this feature. Read this before extending it — it
captures the architecture, the non-obvious edge cases, the bugs that took several
iterations to fix, and concrete guidance for future work. Treat it as the
"why it's built this way" companion to `requirements.md` / `design.md`.

---

## 1. Architecture in one paragraph

A portrait kiosk SPA (React + Vite + TS) that lets a signed-in visitor capture a
webcam photo, pick one of 12 effects, and get an AI-transformed image from an
**existing** FLUX.2 [klein] 9B SageMaker **async** endpoint (`flux2-klein-9b-g6e2`).
**There is NO backend API.** The browser calls AWS directly (AWS SDK v3) using
**temporary credentials from a Cognito Identity Pool**, authorized by IAM roles.
Server-side compute is minimal: one **Scheduler Lambda** that starts/stops the
endpoint on a timetable, and one **Invoke_Proxy Lambda** (behind an
IAM-authenticated Function URL) that performs the single `InvokeEndpointAsync`
call the browser cannot make directly (the SageMaker runtime API has no CORS —
see §3.14). Everything is provisioned by one CDK stack
(`AiPhotoBoothStack`). Three packages at repo root: `ui/`, `backend/`, `cdk/`.

```
Browser (SPA)  ──Cognito sign-in──>  User Pool
   │  GetCredentialsForIdentity (ID token)
   ├──> Identity Pool ── group "admin" ? Admin_Role : Authenticated_Role
   │
   ├─ S3 PutObject (inputs/) + InvokeEndpointAsync          (generation)
   ├─ S3 Head/GetObject (outputs/ , failures/)              (poll)
   ├─ SES v2 SendEmail (raw MIME, PNG attached)             (email my photo)
   ├─ SageMaker List/Describe/Create/DeleteEndpoint  [admin]
   ├─ DynamoDB Query/Put/Delete on Schedule_Store    [admin]
   └─ Cognito ListUsers / AdminUserGlobalSignOut     [admin]

EventBridge (2x rate(1 min) rules) ──> Scheduler Lambda ──> DynamoDB read + SageMaker reconcile
```

## 2. Package / file map (where things live)

- `ui/src/config.ts` — runtime config accessor. Reads `window.__BOOTH_CONFIG__`
  (injected as `config.js` in prod) with `VITE_*` env fallback for local dev.
- `ui/src/auth/authService.ts` — Cognito session + Identity-Pool STS creds, silent
  refresh. `useAuth.ts` — React hook exposing `isAuthenticated`, `isAdmin`,
  `connectionStatus`.
- `ui/src/api/awsClients.ts` — lazy SDK v3 clients + `credentialsProvider` +
  `withAuthRetry` + `isAuthError`.
- `ui/src/api/{generation,endpoints,schedule,users,email}.ts` — the direct-to-AWS
  modules. Pure helpers duplicated for the UI: `requestBuilder.ts`,
  `pollDecision.ts`, `statusMap.ts`, `workingHours.ts`, `timezone.ts`.
- `ui/src/booth/*` — capture-flow state machine (`machine.ts`) + screens.
  Start/Camera/Review are individual screens; **Effects/Loading/Result/Error are
  rendered by one combined `StudioView`** (see below) so the photo, the effect
  options, and the result are co-present. `useLayoutMode.ts` picks monitor vs
  mobile; `ImageCarousel.tsx` is the mobile original/generated toggle.
- `ui/src/admin/*` — EndpointPanel, ScheduleCalendar, UsersPanel, AdminTab.
- `backend/src/lib/*` — shared pure logic (effects, request-builder, status-map,
  working-hours, poll-decision, authz). Property-tested with fast-check.
- `backend/src/scheduler/apply.ts` — the only Lambda.
- `cdk/lib/*` — auth, hosting, data, scheduler, email, initial-users, ui-deployment,
  photo-booth-stack (root).

> **Duplication note:** `ui/src/api/*` re-implements several `backend/src/lib/*`
> modules (effects, request-builder, poll-decision, status-map, working-hours).
> This is deliberate: the UI is type-checked in isolation (`tsconfig.app.json`
> `include: ["src"]`, built with `tsc -b`) and **cannot import `backend/src`**
> (TS6307 composite-project error). If you change the catalog, prompts, clamps,
> status mapping, or HH:mm logic, **change BOTH copies** and keep them identical.

## 3. The edge cases & bugs that cost iterations

### 3.1 Cognito `custom:profile` custom attribute — DROPPED, use the group
Three separate deploy failures came from a custom user-pool attribute:
- `Type for attribute {custom:profile} could not be determined` on AdminCreateUser
  (even with `StringAttribute({minLen,maxLen})` constraints).
- A deferred `AdminUpdateUserAttributes` then failed with `Attribute does not
  exist in the schema` (schema-propagation race).

**Resolution: removed the custom attribute entirely.** Admin identity is conveyed
**solely by membership in the Cognito `admin` group**, which:
1. the Identity Pool maps to `Admin_Role` (the real authorization), and
2. Cognito injects as `cognito:groups` in the ID token (the UI's cosmetic toggle).

`isAdmin(claims)` checks BOTH `custom:profile === "ADMIN"` OR `cognito:groups`
includes `admin`, so the group path lights up the Admin tab with no code change.
**Lesson:** for admin/role signals, prefer Cognito groups over custom attributes —
groups are reliable to write at create time and need no schema gymnastics.

### 3.2 SES identity "already exists"
`AWS::SES::EmailIdentity` create fails if the email is already a verified identity
in the account. Added `useExistingIdentity` (context flag
`senderEmailAlreadyVerified=true`) so the construct **references** the deterministic
identity ARN instead of creating it. The IAM `ses:SendEmail` grant uses the ARN
either way, so referencing is fully sufficient.

### 3.3 `global is not defined` (white screen, local dev)
`amazon-cognito-identity-js` (via `buffer`) references Node's `global`, absent in
the browser. Fixed in `vite.config.ts`: `define: { global: "globalThis" }`. Needed
for both dev and prod build.

### 3.4 Login "failed" — username case sensitivity
Cognito usernames are **case-sensitive**. Users were created lowercase
(`admin`, `visitor`); logging in as `Admin`/`Visitor` fails. Always lowercase.

### 3.5 Initial users start in FORCE_CHANGE_PASSWORD
`AdminCreateUser` leaves users unable to sign in until a permanent password is set:
```
aws cognito-idp admin-set-user-password --user-pool-id <id> \
  --username admin --password 'Aws@2015' --permanent --region us-east-1
```
Quote passwords (the `@`). Verify with `admin-get-user ... --query UserStatus` →
expect `CONFIRMED`.

### 3.6 EndpointPanel showed "No endpoints available" with no way to start
`ListEndpoints` only returns endpoints that **currently exist**. A deleted booth
endpoint (`NOT_DEPLOYED`) isn't listed, so there was no row to act on. **Fix:** the
panel always merges the configured managed endpoint (`config.endpointName`) into the
list via `DescribeEndpoint` and auto-selects it, so Start is always reachable.

### 3.7 The "scheduler didn't start at 10pm" non-bug — it was AM/PM data
The scheduler was healthy. The DynamoDB item stored `10:00`–`11:00` (10 **AM**),
not 22:00. The HTML time picker captured AM. Diagnosed by reading the live state:
`aws dynamodb scan` (the item), Lambda env (`TIMEZONE=Asia/Jerusalem`),
`aws events list-rules` (both ENABLED), and the Lambda log group's reconcile lines
(`time: '23:50'`, correct Jerusalem +3). **Lesson:** before "fixing" a scheduler,
dump the actual stored data + the Lambda's computed wall-clock from logs.

### 3.8 Timezone: browser vs scheduler must agree
The scheduler computes "now" in `Asia/Jerusalem` via `Intl.DateTimeFormat`. The UI
must use the **same** timezone (not the kiosk's OS tz) when it computes "today" or
writes a manual-start window — otherwise off-by-hours/day. Added
`ui/src/api/timezone.ts` (`computeWallClock`, `nowInScheduleTz`) mirroring the
scheduler, and threaded `timezone` through runtime config (`VITE_TIMEZONE` /
`config.js`).

### 3.9 Manual start vs the scheduler (the "it shuts down what I started" trap)
If you manually Start outside any window, the next reconcile sees IN_SERVICE +
outside-window → `DeleteEndpoint`. **Fix:** `startEndpoint()` writes/extends today's
Working_Hours to cover `[now, now+20min)` in scheduler tz, **unioning** with any
existing window so it never clobbers the admin's real schedule.

### 3.10 vitest v4 quirk: persistent rejections
`mockRejectedValue(...)` (persistent) is flagged as an unhandled rejection even when
the code catches it; tests fail spuriously. Use `mockRejectedValueOnce(...)` or a
`try/catch` + `expect(caught)` instead.

### 3.11 HTML `<input type="email">` blocks custom validation
Native constraint validation prevents form submit on a bad address before your
handler runs, so your friendly error never shows. Use `type="text"
inputMode="email"` and validate in JS.

### 3.12 `tsc -b` (project-reference build) checks test files too
`noUnusedLocals` will fail the production build on an unused import **in a test
file**. Keep test imports clean; `npm run build` runs `tsc -b && vite build`.

### 3.13 S3 Body type in the browser SDK
`GetObject` Body in the browser is a web stream. Use `transformToByteArray()`, copy
into a fresh `Uint8Array` (the SDK's may be typed over `ArrayBufferLike`, which trips
`BlobPart` typing), then `new Blob([copy], { type: "image/png" })`.

### 3.14 SageMaker runtime API has NO CORS — InvokeEndpointAsync must be proxied
This is the big one. **S3 supports configurable CORS, but the SageMaker *runtime*
API (`runtime.sagemaker.<region>.amazonaws.com`) does not.** It never returns
`Access-Control-Allow-Origin` and doesn't answer the browser preflight, so a
browser call to `InvokeEndpointAsync` is **always** blocked by CORS — no bucket
or account setting can fix it. (The earlier S3 PutObject CORS error WAS fixable
on the bucket; this one is not.)

**Fix: proxy only that one call.** Input upload (S3 PutObject) and polling
(S3 Head/GetObject) stay browser-direct (S3 has CORS). `InvokeEndpointAsync` goes
through a tiny Lambda:
- `backend/src/invoke/handler.ts` — SDK v3 `@aws-sdk/client-sagemaker-runtime`,
  validates `inputLocation` is in the configured bucket + inputs prefix, calls
  `InvokeEndpointAsync`.
- `cdk/lib/invoke-construct.ts` — `NodejsFunction` + **Function URL with
  `authType: AWS_IAM`** (NOT public) + native CORS (CloudFront origin +
  localhost). Lambda role: only `sagemaker:InvokeEndpointAsync` on the endpoint
  ARN. Caller roles get `lambda:InvokeFunctionUrl` (scoped, with the
  `lambda:FunctionUrlAuthType=AWS_IAM` condition) via `grantInvoke()`.
- `ui/src/api/invokeProxy.ts` — SigV4-signs (`@smithy/signature-v4`,
  `@smithy/protocol-http`, `@aws-crypto/sha256-js`; service `lambda`) a `fetch`
  POST to the Function URL using the Identity-Pool creds from
  `credentialsProvider`. `generation.ts` calls this instead of the runtime SDK.
- Runtime config gains `invokeFunctionUrl` (stack output `InvokeFunctionUrl`,
  `VITE_INVOKE_FUNCTION_URL` for local dev). **Set it after deploy** or local
  generation fails with `Missing runtime config: "invokeFunctionUrl"`.
- Why Function URL + IAM (not API Gateway): the architecture forbids API
  Gateway; a Function URL with `AWS_IAM` is non-public, supports CORS (the thing
  the runtime API lacks), and reuses the existing Cognito credential chain.
- **Lesson:** AWS service API endpoints generally don't support CORS. Only S3
  (and a few others) do. Anything else the browser "can't reach" needs a signed
  Function URL / Lambda proxy, never a CORS toggle.

### 3.15 Function URL `AWS_IAM` needs BOTH `InvokeFunctionUrl` AND `InvokeFunction`
After §3.14 the browser reached the Function URL but every call 403'd
("Forbidden. For troubleshooting Function URL authorization issues..."), and it
bounced to the sign-in screen (because `isAuthError` treats 403 as auth-expiry
and `withAuthRetry` gives up after one refresh). **The caller's IAM policy must
grant BOTH `lambda:InvokeFunctionUrl` AND `lambda:InvokeFunction` on the
function** — granting only `InvokeFunctionUrl` (the action whose name matches
the operation) is **insufficient and yields 403**.
- Verified empirically by attaching `AWSLambda_FullAccess` (worked → it's a
  permission gap, not signing), then bisecting: `InvokeFunctionUrl` alone → 403;
  `InvokeFunctionUrl` + `InvokeFunction` (both scoped to the function ARN) →
  reached the function (502 only because the endpoint was down).
- Red herrings ruled out along the way: the SigV4 signing was correct the whole
  time (a Node repro with the same `@smithy/signature-v4` libs returned 502 with
  admin creds); a `lambda:FunctionUrlAuthType=AWS_IAM` **condition on the
  identity statement** is harmful (the key isn't reliably in the identity-eval
  context → `implicitDeny`); a resource-based permission is **not** needed
  same-account. The fix is purely the two-action identity grant in
  `cdk/lib/invoke-construct.ts` `grantInvoke()`.
- An admin/`lambda:*` principal masks this bug — always test with the actual
  scoped Cognito role, not your developer admin creds.
- **Debugging recipe:** reproduce the exact browser path in Node — SRP sign-in
  via `amazon-cognito-identity-js`, mint creds via `fromCognitoIdentityPool`,
  sign with `@smithy/signature-v4`, `fetch` the URL — and read the real status.
  403 = authorizer/permissions; 502 = reached the function (e.g. endpoint down);
  200 = full success.

## 4. Scheduler deep-dive (the most subtle subsystem)

- **Cadence:** EventBridge minimum granularity is **1 minute**. To approximate 30s
  we run **two** `rate(1 minute)` rules: Rule A fires the Lambda directly; Rule B
  sends `{ delaySeconds: 30 }` and the handler `await`s 30s before reconciling.
  Verified firing ~twice/minute in logs.
  - **Cost caveat:** Rule B's invocation is billed for the full ~30s idle wait
    (logs showed `Billed Duration: 30359 ms`). If cost matters, replace the
    self-delay with EventBridge **Scheduler** (1-min) + accept 1-min granularity,
    or a Step Functions Wait, or just one rule at 1-min. Reconcile is idempotent,
    so coarser cadence only affects latency-to-react, never correctness.
- **Reconcile logic (`reconcile()` in `apply.ts`):**
  - desiredRunning = item exists for today AND `isWithinWindow(now, start, end)`
    (half-open `[start, end)`, fail-closed on malformed data).
  - inside-window + `NOT_DEPLOYED` (`canStart`) → `CreateEndpoint`.
  - outside-window + `IN_SERVICE` (`canStop`) → `DeleteEndpoint`. **Only acts on
    IN_SERVICE** — leaves CREATING/DELETING/FAILED alone so it never thrashes.
  - else → NONE. Idempotent.
- **Per-day windows only.** A window cannot span midnight (HH:mm capped at 23:59).
  Overnight hours would need two items or a date-aware model — not implemented.
- **Timezone source of truth:** the `TIMEZONE` Lambda env var (CDK param
  `SchedulerTimezone`, currently `Asia/Jerusalem`). Items store wall-clock `HH:mm`
  in that tz.

## 5. Auth & credentials (silent refresh contract)

Two independent things expire — the Cognito **token** and the Identity-Pool **STS
creds** — each with its own refresh path (`authService.refreshSession()` and
`refreshCredentials()`). Every AWS call goes through `withAuthRetry()`:
1. run once; 2. on an **auth-flavoured** error refresh session + creds and retry
**once**; 3. second auth failure → `requireSignIn()` → sign-in screen.
- `isAuthError` is true for ExpiredToken/InvalidClientTokenId/NotAuthorized/
  AccessDenied/401/403; **false** for throttling/5xx/network (those are not retried
  here). Don't broaden it to retry throttling — that belongs to the SDK's own retry.

## 6. IAM model (least privilege, browser-exposed)

- **Authenticated_Role** (every signed-in user): `sagemaker:InvokeEndpointAsync`
  (endpoint ARN); `s3:PutObject` (inputs/*); `s3:GetObject` (outputs/*, failures/*);
  `s3:ListBucket` (prefix-scoped); `ses:SendEmail`/`SendRawEmail` (scoped to the
  sender identity + `ses:FromAddress` condition).
- **Admin_Role** = Authenticated_Role PLUS `sagemaker:List/Describe/Create/Delete
  Endpoint`; DynamoDB `Query/PutItem/DeleteItem` on Schedule_Store;
  `cognito-idp:ListUsers` + `AdminUserGlobalSignOut` (scoped to the user pool ARN).
- **No `iam:PassRole`.** The booth only `CreateEndpoint` (by existing config name)
  and `DeleteEndpoint` — neither passes a role. The SageMaker *execution* role
  belongs to the Model/EndpointConfig (made by the notebook), never passed here.
  Don't reintroduce a PassRole param.
- **Security tradeoff (accepted):** real AWS creds reach the browser, bounded only
  by these policies. Fine for a single-kiosk lobby demo behind Cognito; NOT for
  multi-tenant prod (which should reintroduce a server-side authz tier).

## 7. Deploy / operate cheat-sheet

### One-shot deploy (preferred)
Run the repo-root script — it does everything automatically and is re-runnable:
```bash
./deploy.sh
```
It (1) builds `ui/` + `cdk/`, (2) deploys `AiPhotoBoothStack` with all params +
context, (3) reads the stack outputs **including the Invoke_Proxy
`InvokeFunctionUrl`**, (4) **regenerates `ui/.env.local`** from those outputs so
`npm run dev` works against the fresh resources (no hand-copying the Function
URL or any id), and (5) sets PERMANENT passwords for the initial admin/visitor
users. Override any value via env vars, e.g.
`SENDER_EMAIL=me@example.com ENDPOINT_NAME=... ./deploy.sh`.

### Manual equivalent (what the script runs)
```bash
# build SPA first so the BucketDeployment picks up ui/dist
( cd ui && npm run build )
( cd cdk && npm run build )

cd cdk
npx cdk deploy \
  --context existingIoBucketName=sagemaker-us-east-1-346399954218 \
  --context senderEmailAlreadyVerified=true \
  --parameters SenderEmail=michshap@amazon.com \
  --parameters EndpointName=flux2-klein-9b-g6e2 \
  --parameters EndpointConfigName=flux2-klein-9b-g6e2 \
  --parameters InitialAdminUsername=admin \
  --parameters InitialStandardUsername=visitor \
  --parameters SchedulerTimezone=Asia/Jerusalem
```
- `existingIoBucketName` MUST be the bucket the FLUX endpoint writes to
  (`AsyncInferenceConfig.OutputConfig.S3OutputPath`) — the browser reads/writes there
  directly. Default `sess.default_bucket()` = `sagemaker-<region>-<account>`.
- Required param with no default: `SenderEmail`.
- A failed/rolled-back `CREATE` can't be updated — `cdk destroy` then re-deploy.
- After deploy: set user passwords (§3.5); verify the SES sender link if newly
  created; open the `DistributionDomainName` output.
- **SES sandbox:** until production access, recipients must also be verified.

### Run locally
`ui/.env.local` holds `VITE_*` mirrors of the stack outputs (incl. `VITE_TIMEZONE`).
`npm run dev` → http://localhost:5173, hitting the real deployed AWS resources.
- `config.js 404` locally is harmless (env fallback used).
- S3 from `localhost` may need a CORS rule on the I/O bucket. **This is now
  managed by CDK** via `cdk/lib/io-bucket-cors-construct.ts`
  (`IoBucketCorsConstruct`): an `AwsCustomResource` calls `s3:PutBucketCORS` on
  the referenced shared bucket at deploy time, wiring in the freshly-created
  CloudFront distribution domain automatically (so you never hand-copy the
  domain). It writes BOTH the original SageMaker rule AND the booth rule
  (CloudFront origin + `http://localhost:5173`). It has **no onDelete** — on
  stack delete it leaves the shared bucket's CORS untouched (the bucket is
  pre-existing/shared; clobbering it would break SageMaker Studio).
  - PutBucketCors **replaces** the whole CORS config, so both rules must be
    written together in the construct.
  - Manual fallback (e.g. if the construct isn't deployed yet) still works:
    ```
    aws s3api put-bucket-cors --bucket <io-bucket> --region us-east-1 \
      --cors-configuration file://cdk/io-bucket-cors.json
    ```
  (Additive change to the shared SageMaker bucket — flag it.)

### Live-state troubleshooting (do this BEFORE changing code)
```bash
# What did the UI actually store?
aws dynamodb scan --table-name <ScheduleTable> --region us-east-1
# Is the scheduler's tz right?
aws lambda get-function-configuration --function-name <fn> --query Environment.Variables
# Are the rules firing?
aws events list-rules --query "Rules[?contains(Name,'Scheduler')].[Name,ScheduleExpression,State]"
# What did the scheduler decide (and at what wall-clock)?
aws logs describe-log-groups --query "logGroups[?contains(logGroupName,'Scheduler')].logGroupName"
aws logs tail <log-group> --since 15m | grep -iA8 "reconcile decision"
```

## 8. Testing conventions

- Pure logic (`backend/src/lib/*`, `ui/src/booth/machine.ts`) → **property tests**
  with `fast-check`, `{ numRuns: 100 }`, one property per file, tagged
  `// Feature: ai-photo-booth, Property {n}: ...`.
- UI components → React Testing Library; **mock the `api/*` modules** so tests don't
  instantiate the AWS client chain (which reads runtime config and throws). When you
  add a UI module that imports `api/*` or `config`, add the mock to the test or you'll
  get `Missing runtime config: "region"` at import time.
- CDK → `aws-cdk-lib/assertions` `Template.fromStack`. Key invariants asserted:
  Identity Pool + both roles + role attachment, **zero** API Gateway resources,
  `DeletionPolicy: Delete` everywhere, `autoDeleteObjects`, two EventBridge rules,
  SES identity, `AiPhoto` tag, `ses:SendEmail` granted, **no `iam:PassRole`**.
- `Tags.of(stack).add("AiPhoto", "true")` tags every taggable resource.

## 8a. Responsive capture studio (monitor vs mobile)

The capture phase has TWO layouts driven by `ui/src/booth/useLayoutMode.ts`
(breakpoint `MONITOR_MIN_WIDTH_PX = 820`):
- **monitor** (large portrait screen): background options stacked on the LEFT,
  character options on the RIGHT, original photo centered, generated image
  directly BELOW it.
- **mobile** (phone): image area on top with an `ImageCarousel` (Original /
  Generated tabs), option groups stacked BELOW. The generated image replaces
  the original in-place; a new generation replaces the previous one.

Key design points:
- **One component, `StudioView.tsx`, renders the Effects/Loading/Result/Error
  phases together.** The pure FSM (`machine.ts`) is still the source of truth;
  `BoothFlow` maps the machine state → a `StudioPhase` (`idle|loading|result|
  error`) + photo + generatedUrl and hands it to `StudioView`. The old
  `EffectSelector`/`LoadingScreen`/`ResultScreen`/`ErrorScreen` were deleted.
- **Regenerate in place:** the FSM `Result` and `Error` states now accept
  `SELECT` (→ `Loading` with the retained `capturedPhoto`), so picking another
  effect from the result view regenerates without a full New Session. `Result`
  therefore carries `capturedPhoto` now (property tests 5 & 7 updated to match).
  `BoothFlow` revokes the previous object URL before each new generation to
  avoid leaking blobs.
- The app shell caps width at `--portrait-max-width` (480px); a
  `@media (min-width: 820px)` rule lifts that cap so the three-column monitor
  studio has room (the studio caps + centers its own content).
- The booth no longer uses `KioskScreen` for the studio phase — it owns its own
  responsive grid/flex root. Start/Camera/Review still use `KioskScreen`.

### 3.16 Camera capture must mirror to match the live preview
The webcam preview is mirrored via CSS (`.camera-view__video { transform: scaleX(-1) }`)
so it feels like a mirror. The canvas capture must mirror too (`ctx.translate(w,0);
ctx.scale(-1,1)` before `drawImage`), or the saved photo is flipped left/right
relative to what the visitor saw. (Test canvas mocks need `translate`+`scale` stubs.)

### 3.17 SES send IAM scope — identity AND configuration-set, not just sender
`ses:SendEmail`/`SendRawEmail` authorizes against EVERY resource the message
touches: the sender identity, the **recipient** identity, AND any default
**configuration set** attached to the sender. Scoping the policy `resources` to
only the sender identity ARN yields a 403 `AccessDeniedException` (on the
recipient identity, then on `configuration-set/<name>`). Fix: grant the action on
`identity/*` AND `configuration-set/*` in-account, locked down with a
`ses:FromAddress` StringEquals condition so the visitor can only send FROM the
booth's verified address. Recipient restriction is enforced by SES sandbox, not
IAM. See `auth-construct.ts` `sesSendStatement()`.

### 3.18 A failed email must not sign the visitor out
A generic 403 routes through `withAuthRetry` → `requireSignIn()` (sign-in screen).
Email failures (sandbox `MessageRejected`, IAM `AccessDenied`) are delivery
problems, not expired sessions. `sendPhotoEmail` no longer uses `withAuthRetry`;
it catches SES errors and rethrows a friendly `EmailDeliveryError`, so a bad
recipient shows an inline message instead of bouncing to login. **Gotcha:** any
test that `vi.mock("../api/email", ...)` must also export `EmailDeliveryError`
(StudioView does `err instanceof EmailDeliveryError`; an undefined export throws
inside the catch).

### 3.19 SES sandbox: recipients must be verified
Account is in SES sandbox (`ProductionAccessEnabled: false`). Only verified
identities can receive (`michshap@amazon.com`, `michshap+1@amazon.com`). Sending
to an arbitrary address (e.g. a gmail) fails with `MessageRejected`. To deliver
to anyone, request SES production access, or verify the recipient first.
**Update:** the self-service production-access request was **auto-DENIED**
(CaseId 178073209400306) on this internal account — SES production is likely
blocked by org policy here. Options: appeal the case in the SES console, use a
different AWS account, or skip email. The booth therefore always offers a
**Download photo** button (`<a download>` of the result object URL) as a
no-SES-needed fallback, so visitors can keep their photo regardless of email.

### 3.20 FLUX.2 prompt: stop it inventing extra people
Editing a single-subject selfie with a "person" effect, FLUX.2 tended to ADD
extra costumed people (turning a portrait into a themed group shot). Fixes that
worked, in order of impact:
  1. **Lead with the composition lock.** FLUX.2 weights earliest tokens most, so
     each person prompt now STARTS with "Keep the exact same people from the
     original photo — the same number of people, same faces, same positions; do
     not add, remove, or duplicate anyone." THEN the costume.
  2. Frame the edit as "change only their clothing: dress each existing person
     as ..." rather than "dress every person as ..." ("every person" reads as
     "make it a crowd").
  3. Keep identity/gender/age/skin tone preservation at the end.
See `effects.ts` (both copies). If it still adds people for a given effect,
push the count lock even harder or lower guidance.

## 8b. Admin cost panel (Cost Explorer by AiPhoto tag)

The admin Schedule tab shows spend for the booth's `AiPhoto`-tagged resources:
- **Month-to-date** total near the calendar header.
- **Per-day breakdown by AWS service** + the day total, shown below the time
  editor when a date is selected.
- A clear "costs can take up to 24h to appear" note (Cost Explorer data lags).

How it works:
- `ui/src/api/cost.ts` calls **Cost Explorer `GetCostAndUsage`** (AWS SDK v3,
  `@aws-sdk/client-cost-explorer`) filtered by `Tags: { Key: "AiPhoto",
  Values: ["true"] }`. DAILY granularity grouped by `SERVICE` for the per-day
  breakdown; MONTHLY for month-to-date. `End` is exclusive (query `[day, day+1)`).
- Cost Explorer is **global** — the client is pinned to `us-east-1` regardless
  of stack region (`getCostExplorerClient()`).
- Admin_Role granted `ce:GetCostAndUsage` on `*` (CE has no resource-level
  scoping). Read-only.
- **Cost allocation tag activation:** `cost-allocation-tag-construct.ts` calls
  `ce:UpdateCostAllocationTagsStatus` via an AwsCustomResource to activate
  `AiPhoto`. This only succeeds on the **management/payer account** and only
  after tagged resources have reported usage (≤24h); it `ignoreErrorCodesMatching`
  AccessDenied so a member-account deploy still succeeds. If the tag isn't
  active, the panel shows zero/empty — activate it once in the Billing console.
  (On this account it activated successfully — `Status: Active`.)
- All amounts use `UnblendedCost`; `formatCost()` renders the currency.

## 8c. On-screen email keyboard (kiosk monitor)

The kiosk monitor has no physical keyboard, so the email field can't rely on a
native one. In the **monitor** layout the email `<input>` is `readOnly` and
tapping it opens a minimal on-screen keyboard (`ui/src/booth/EmailKeyboard.tsx`):
letters + digits + only email-legal symbols (`@ . _ - +`), domain shortcuts
(`@gmail.com`, `.com`, …), backspace, Clear, Done. In the **mobile** layout the
native OS keyboard is used instead (no on-screen keyboard rendered).

Validation: input from BOTH paths is regex-sanitized (`sanitizeEmail()` strips
anything outside `[a-z0-9@._+-]`, lowercased) so invalid characters can never be
entered, and the existing `isValidEmail()` regex still gates submit. The form is
controlled, so the keyboard never owns the value — it just calls
`onInput/onBackspace/onClear/onDone`.

## 8d. Selectable endpoint config + "current" pointer (no hardcoded config)

The booth no longer hardcodes one endpoint config. The admin Endpoints panel
lists the account's SageMaker **endpoint configurations** (`ListEndpointConfigs`),
stacked one below the other, each with independent Start/Stop. Exactly one is
the **current** config, used by generation and the scheduler.

- **Current pointer:** a single DynamoDB item in the Schedule_Store —
  `pk = sk = "CONFIG#CURRENT"`, attr `configName`. A fixed key means only one
  config can be current; writing a new value overwrites the old (selecting B
  un-currents A). Helpers: `getCurrentConfigName` / `setCurrentConfigName`.
- **Managed (curated) list:** the admin doesn't manage ALL account configs —
  they ADD configs to a curated list stored in DynamoDB (`pk = "CONFIG#MANAGED"`,
  `sk = "CONFIG#<name>"`). The Endpoints panel shows an autocomplete picker of
  ALL account configs (`ListEndpointConfigs` → `listAllConfigNames`, rendered as
  a `<datalist>`); clicking **Add** writes a managed item. Only managed configs
  get Start/Stop/Make-current/Remove rows (`listManagedConfigsWithStatus`). The
  first managed config auto-becomes current when none is set. Helpers in
  `ui/src/api/schedule.ts`: `listManagedConfigs` / `addManagedConfig` /
  `removeManagedConfig`.
- **Endpoint name === config name.** Starting a config calls
  `CreateEndpoint(EndpointName=cfg, EndpointConfigName=cfg)`; stopping deletes
  that endpoint. A config's row status is the live status of that endpoint.
- **Generation** resolves the current config SERVER-SIDE: the invoke proxy
  Lambda reads the `CONFIG#CURRENT` pointer (DynamoDB `GetItem`) and invokes
  that endpoint, falling back to `ENDPOINT_NAME` env when unset. So the browser
  never needs to know which is current.
- **Scheduler** resolves the current config the same way and manages that
  endpoint. **Working_Hours are now booth-wide** (keyed `ENDPOINT#booth`,
  `BOOTH_SCHEDULE_NAME`), decoupled from the changing config name, so switching
  the current config never orphans the schedule.
- **IAM:** admin gets `sagemaker:ListEndpointConfigs` + Create/Delete/Describe on
  `endpoint/*` (any name) + DynamoDB `Query/GetItem/PutItem/DeleteItem` on the
  table (**GetItem is required** to read the `CONFIG#CURRENT` pointer — omitting
  it makes the panel 400 → `isAuthError` → bounce to login). Invoke Lambda gets
  `InvokeEndpointAsync` on `endpoint/*` + `dynamodb:GetItem` on the table.
  Scheduler manages `endpoint/*`.
- The `EndpointConfigName` CfnParameter / runtime-config field remain as
  fallbacks only; the live selection is the DynamoDB pointer.

## 9. Extension ideas / where to be careful

- **Reference image must be RAW base64 — strip the `data:` URI prefix.** The
  webcam capture is a full data URL (`data:image/png;base64,iVBOR...`); the
  endpoint's `inference.py` decodes with `base64.b64decode(..., validate=True)`,
  which rejects the `data:image/png;base64,` prefix ("Only base64 data is
  allowed" → failure record). `buildAsyncRequest` (both `ui/src/api/` and
  `backend/src/lib/` copies) runs the photo through `stripDataUriPrefix()` before
  putting it in `images[]`. Symptom if regressed: generation always writes a
  `<uuid>-error.out` failure record with a base64 traceback. (Note: a too-tiny
  reference image can also crash the worker — "Worker died" — so test with a real
  webcam-sized photo, not a 2x2 px stub.)

- **Async output key is server-generated — poll the returned `OutputLocation`.**
  SageMaker async inference does NOT name the result after the input key. The
  result lands at a server-chosen UUID key (e.g. `flux2-klein-outputs/<uuid>.out`),
  returned by `InvokeEndpointAsync` as `OutputLocation` (and `FailureLocation`).
  The Invoke_Proxy returns both; `submitGeneration` parses the keys out of those
  URIs into `SubmitResult.outputKey`/`failureKey`, and `pollGeneration` HEADs
  those exact keys. Do NOT derive the poll key from the jobId/input key — that
  was a bug that made the booth poll `booth-<id>.json.out` forever (404 loop).

- **Result image moderation** was deliberately removed (no Rekognition). If you
  re-add NSFW screening, it must happen client-side before display OR move generation
  behind a Lambda — the browser-direct model has no server hook to gate on.
- **Overnight schedules / recurring weekly schedules:** current model is one item
  per `ENDPOINT#name` / `DAY#YYYY-MM-DD`, single window, no midnight span. Extend the
  item shape + `isWithinWindow` together, and keep the UI/scheduler copies in sync.
- **Multiple managed endpoints:** the schema already partitions by endpoint
  (`pk = ENDPOINT#<name>`); the UI hardcodes one `config.endpointName`. Generalize the
  EndpointPanel/scheduler to iterate endpoints if needed.
- **Stronger admin auth:** the Admin tab is cosmetic; IAM is the boundary. Anyone in
  the `admin` group has the Admin_Role. Manage membership via Cognito groups.
- **Cost:** the g6e endpoint is the expensive part; the scheduler exists to keep it
  off outside hours. The Rule-B 30s self-delay is the only notable Lambda cost
  (see §4). CloudFront/S3/DynamoDB/Cognito are negligible at kiosk scale.
- **Bundle size:** the SPA is ~570 kB (AWS SDK v3). Acceptable for a kiosk; if you
  care, code-split the admin/SES clients behind dynamic `import()`.
- **`AdminUserGlobalSignOut` is not instant:** it revokes refresh tokens, but an
  already-issued access token stays valid until expiry (≤1h). There's no instant
  kill-switch for an in-flight access token.

## 10. Golden rules (summary)

1. Two copies of shared logic (ui + backend) — change both.
2. Admin = Cognito **group**, never a custom attribute.
3. Browser and scheduler must use the **same configured timezone**.
4. Before "fixing" the scheduler, read DynamoDB + Lambda logs first.
5. Mock `api/*` in UI tests; keep test imports clean (`tsc -b` checks them).
6. No `iam:PassRole`; no API Gateway; tag everything `AiPhoto`.
7. Manual Start must write a protective Working_Hours window or the scheduler
   will stop it.

## 11. Lessons learned & tips for future development

Meta-lessons from building this, beyond the per-bug entries in §3. Read this
before extending — it captures the *recurring shapes* of the problems, the parts
that were hardest to get right, and the debugging moves that actually worked.

### 11.1 The "redirect to login for no reason" trap (recurring, cost the most time)
Almost every confusing failure in this project ended with the app bouncing to the
sign-in screen, which hid the real cause. Why: any AWS call that returns **400/403
AccessDenied** flows through `isAuthError → withAuthRetry → requireSignIn()`. So a
plain **missing IAM permission** looks identical to an expired session.
- This bit us at least three times: missing `dynamodb:GetItem` (current-config
  read), the Function URL two-action gap (§3.15), and SES recipient/config-set
  scope (§3.17). Each presented as "logs in, flashes, redirects."
- **Debugging move that works:** open the browser Network tab, find the failing
  request, read the actual service + status + error message in the response body.
  Don't trust the redirect — it's a symptom, not the cause.
- **When adding ANY new browser→AWS call, add its IAM action in the SAME change.**
  The single most common mistake here was wiring a new SDK call without the
  matching grant. Grep the role policies after, or expect the login bounce.
- Consider (future): make `isAuthError` distinguish "AccessDenied with valid
  creds" (a permission bug — surface it) from "ExpiredToken/NotAuthorized" (real
  re-auth). Right now both re-auth, which masks permission bugs.

### 11.2 Browser-direct-to-AWS: what works and what doesn't
The no-backend architecture is elegant but has sharp edges discovered the hard way:
- **CORS is the gating constraint.** S3 supports CORS (configurable per-bucket);
  most AWS service APIs do NOT (SageMaker runtime, §3.14). Before assuming "the
  browser can call X", check whether X's endpoint returns CORS headers. If not,
  it needs a signed Lambda Function URL proxy — there is no client-side fix.
- **Tag every browser-reachable action with its blast radius.** Real STS creds
  reach the browser; the IAM policy is the *only* boundary. Keep resource ARNs
  scoped (prefixes, single table) and never grant `*` actions to the
  Authenticated_Role.
- **Test with the SCOPED role, never your admin creds.** A `lambda:*` / admin
  principal masks missing-permission bugs (§3.15). The Node repro recipe
  (SRP sign-in → `fromCognitoIdentityPool` → real SDK call) reproduces the exact
  visitor path and is the fastest way to confirm an IAM fix without a redeploy.

### 11.3 Two copies of shared logic — the tax you keep paying
`ui/src/api/*` duplicates `backend/src/lib/*` because `tsc -b` can't import across
the composite-project boundary (TS6307). This caused silent drift bugs (a prompt
or key changed in one copy only). Tips:
- Treat the pair as one unit: change both in the same edit, every time.
- The keys/prefixes that MUST match across ui + backend + CDK: the S3
  inputs/outputs/failures prefixes, the DynamoDB `pk`/`sk` shapes
  (`ENDPOINT#booth`, `CONFIG#CURRENT`, `CONFIG#MANAGED`), and the `AiPhoto` tag.
- If this grows, the right fix is a tiny shared package built to JS that both
  consume — not more hand-copying. Deferred because the surface is still small.

### 11.4 Hardest things to implement (where to budget time)
- **Function URL IAM auth (§3.15):** by far the most time-consuming. The action
  name `lambda:InvokeFunctionUrl` is necessary-but-not-sufficient; you also need
  `lambda:InvokeFunction`. Non-obvious and the error (403) is generic.
- **SES from the browser (§3.17, §3.19):** the IAM resource scope (sender +
  recipient + config-set) is unintuitive, AND sandbox blocks arbitrary
  recipients, AND production access was org-denied. Net: email is best-effort;
  the **Download photo** fallback is what actually ships value. For future
  "deliver the image" features, prefer download / pre-signed URL / QR over email
  unless production SES is confirmed available on the target account.
- **FLUX.2 prompt control (§3.20):** instruction-based image editing is finicky.
  Lead with the constraint you care most about (word order matters), phrase
  preservation positively (no negatives), and test on real group photos — bugs
  only show with >1 subject.

### 11.5c Dependency compatibility matrix (the painful part)
**`torch` is NEVER in `requirements.txt`.** It's the DLC's CUDA-matched build and
the only arch-specific piece (g6e=2.5/sm_89, g7e=2.7/sm_120). A pip torch
overwrites it → "no kernel image" crash. Rules that keep cold starts sane:
- **Pin exact** where a behavior depends on it: `diffusers==0.38.0` (first
  Flux2KleinPipeline release; pure-Python, torch-agnostic),
  `transformers==4.56.2` (the ONE version in `[4.51,5.0)` that has Qwen3 AND
  works on both torch 2.5 and 2.7 — 5.x needs `torch.float8_e8m0fnu` (torch≥2.7)
  so it breaks g6e).
- **Cap the majors** on the rest so an unbounded `>=` can't silently resolve a
  breaking version on a future cold start: `accelerate>=1.0.0,<2.0.0`,
  `protobuf>=4.25,<6`, `safetensors>=0.4.3,<1.0`, `sentencepiece>=0.2.0`.
- The SageMaker toolkit runs `pip install -r requirements.txt` **without
  `--upgrade`** every cold start — so unbounded ranges are a latent time bomb.
- `_ensure_dependencies()` re-pins transformers at runtime with **`--no-deps`**
  so it can never pull a different torch/tokenizers; it NEVER touches torch.
- Everything in requirements is pure-Python / CPU-side, so the same file works
  on both g6e and g7e — only the DLC `framework_version` differs (selection cell).

### 11.5 SageMaker async + the endpoint lifecycle
- **The async output key is server-generated** (a UUID returned by
  `InvokeEndpointAsync`), NOT derived from your input key. Poll the returned
  `OutputLocation`/`FailureLocation`, never a key you constructed. (This was a
  silent 404-forever bug.)
- **Endpoint name === config name** is the convention the whole booth relies on:
  `CreateEndpoint(EndpointName=cfg, EndpointConfigName=cfg)`. The notebook must
  therefore **pin** `endpoint_config_name=endpoint_name` — the SDK otherwise
  auto-generates a timestamped config name and the booth can't recreate it.
- **Runtime-created endpoints aren't CloudFormation resources**, so they don't
  inherit the stack's `AiPhoto` tag. Tag them explicitly on `CreateEndpoint(Tags=
  [...])` (both the UI start path and the scheduler), which needs
  `sagemaker:AddTags` in the role. Otherwise the cost panel undercounts the
  expensive GPU spend.
- **"Worker died" with a valid request** usually means the input image is too
  small/degenerate for the model (we hit it with a 2×2 test PNG), not a code bug.
  Test with a realistic webcam-sized image.
- **GPU arch must match the PyTorch DLC build (`no kernel image is available`).**
  This is a hardware/toolkit mismatch, not a logic bug. g6e = L40S (Ada, sm_89),
  supported by the PyTorch **2.5** DLC. g7e = RTX PRO 6000 **Blackwell** (sm_120,
  CUDA 12.9), which torch 2.5 has NO kernels for — it crashes on the first CUDA
  op regardless of CPU-offload vs `.to("cuda")`. g7e needs a Blackwell-capable
  DLC (**PyTorch ≥ 2.7, CUDA 12.8+**). The notebook pins `framework_version`/
  `py_version` per family in the selection cell (g6e→2.5/py311, g7e→2.7/py312);
  bump to the newest available PyTorch inference DLC in your region if needed.

### 11.6 Notebook editing — don't corrupt cells programmatically
When editing `.ipynb` cells with a script, **read the source string BEFORE
reassigning it.** A placeholder-then-replace ordering blanked a config cell
(had to restore by hand). Safer: read `"".join(cell["source"])`, do string
replacements, then write `cell["source"] = new.splitlines(keepends=True)`, and
re-validate with `json.load` + a quick cell-by-cell print afterward.

### 11.7 Kiosk UX gotchas (touch monitor, no keyboard)
- **No physical keyboard** → any text input needs an on-screen keyboard on the
  monitor layout (`readOnly` input + custom keys); mobile uses the native one.
- **Idle auto-reset must reset on activity AND never fire mid-generation**, or it
  kicks the visitor out while they wait/type. Listen for pointer/key/touch to
  restart the countdown; skip the timer in the `Loading` state.
- **`PrimaryButton` defaults to `block` (full-width hero).** In any inline/row
  context (add-config row, per-config action row) pass `block={false}` or it
  expands to fill the row and squashes its neighbours. This caused two separate
  "looks bad" rounds — check it whenever a PrimaryButton sits next to others.
- **Mirror the capture** to match the mirrored live preview (§3.16), or the photo
  looks "wrong" to the visitor.

### 11.8 Deploy ergonomics
- `./deploy.sh` is the source of truth for a full deploy (build → deploy → read
  outputs → regenerate `ui/.env.local` → set passwords). Use it; don't hand-run
  the steps unless debugging.
- **Concurrent `cdk` invocations clash on `cdk.out`** ("Other CLIs are reading
  from cdk.out"). If a deploy seems stuck, it may already be running — check
  `aws cloudformation describe-stacks --query 'Stacks[0].StackStatus'` and
  `wait stack-update-complete` rather than launching a second deploy.
- A UI-only change still deploys through CDK (the `BucketDeployment` pushes
  `ui/dist`); there's no separate UI deploy.
