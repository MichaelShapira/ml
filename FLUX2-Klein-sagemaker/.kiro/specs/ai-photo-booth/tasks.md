# Implementation Plan: AI Photo Booth

## Overview

This plan implements the AI Photo Booth as three new TypeScript packages at the repository root — `backend/` (shared pure logic + the single `Scheduler_Function` Lambda), `ui/` (React + Vite SPA that calls AWS directly with the SDK for JavaScript v3), and `cdk/` (AWS CDK app) — alongside the existing, untouched `flux2-klein-sagemaker.ipynb`, `prepare_weights.py`, `README.md`, and `code/`.

There is **no per-request backend API**: no API Gateway and no per-request Lambdas. The browser submits and polls generation directly against S3 + SageMaker Runtime, performs admin endpoint management directly against SageMaker, and reads/writes the schedule directly against DynamoDB — all under temporary AWS credentials vended by a Cognito **Identity Pool** and authorized by **IAM roles** (`Authenticated_Role` / `Admin_Role`). The only server-side compute is the `Scheduler_Function`. There is also **no content moderation** (the poll path's terminal statuses are `READY | FAILED | TIMEOUT`).

The build order is test-driven and incremental:

1. Scaffold the three packages with their test runners (Vitest + `fast-check`) — **already complete** — and remove the now-obsolete moderation module and Rekognition dependency.
2. Build/finish the **pure-logic layer** (`backend/src/lib/*` and `ui/src/booth/machine.ts`), each module paired with a property-based test for the design's Properties 1–13 (`fast-check`, `numRuns: 100`). The effect catalog, status map/guards, authz, working-hours validation/serialization, and poll-decision modules are **already implemented**; their property tests, `request-builder.ts`, `isWithinWindow`, and `machine.ts` are not.
3. Build the browser-side AWS layer: the client auth/credentials layer (`ui/src/auth`) and the direct-to-AWS modules (`ui/src/api/{awsClients,generation,endpoints,schedule}.ts`) with integration tests over mocked AWS SDK v3 clients.
4. Build the React SPA capture flow (auth gate, state-machine-driven screens, 12-option effect selector, loading/result, connection indicator, portrait/touch theme) and the admin tab (endpoint panel, schedule calendar).
5. Build the single `Scheduler_Function` (`backend/src/scheduler/apply.ts`).
6. Build the CDK app (auth/hosting/data/scheduler constructs, initial-users custom resource, `RemovalPolicy.DESTROY` everywhere, `autoDeleteObjects` on stack-created buckets, CfnParameters, private S3 + CloudFront OAC, two EventBridge rules, **no API Gateway**) with assertion/snapshot tests.
7. Wire everything together (UI runtime config + CloudFront/S3 deployment) and run a final verification pass.

Each property-based test sub-task is annotated with its design **Property** number, the requirements clause it validates, must use `fast-check` with `{ numRuns: 100 }`, carry the tag comment `// Feature: ai-photo-booth, Property {n}: {property text}`, and live in its own dedicated test file.

## Tasks

- [x] 1. Package scaffolding and shared-tooling cleanup
  - [x] 1.1 Scaffold the `backend/` package
    - Create `backend/package.json`, `backend/tsconfig.json`, `backend/vitest.config.ts`, and the `backend/src/{scheduler,lib,lib/__tests__}` directory tree
    - Target Node 20 + TypeScript; add `vitest` and `fast-check` as dev dependencies and a `test` script
    - Add AWS SDK v3 clients used by the scheduler (`@aws-sdk/client-sagemaker`, `@aws-sdk/client-dynamodb`) and `aws-sdk-client-mock` for tests
    - _Requirements: 8.1, 15.1, 20.1, 21.6_

  - [x] 1.2 Scaffold the `ui/` package
    - Create the Vite + React + TypeScript skeleton (`ui/package.json`, `ui/tsconfig*.json`, `ui/vite.config.ts`, `ui/index.html`, `ui/src/main.tsx`, and the `ui/src/{booth,auth,admin,api,theme,test}` directories)
    - Add `vitest`, `fast-check`, and React Testing Library as dev dependencies and a `test` script
    - Add the AWS Amplify / Cognito auth dependency and AWS SDK v3 browser clients (`@aws-sdk/client-s3`, `@aws-sdk/client-sagemaker-runtime`, `@aws-sdk/client-sagemaker`, `@aws-sdk/client-dynamodb` + `lib-dynamodb`, `@aws-sdk/credential-providers`)
    - _Requirements: 11.1, 25.1_

  - [x] 1.3 Scaffold the `cdk/` package
    - Create `cdk/package.json`, `cdk/tsconfig.json`, `cdk/cdk.json`, `cdk/bin/app.ts` (empty app entry), `cdk/vitest.config.ts`, and the `cdk/lib/` and `cdk/test/` directories
    - Add `aws-cdk-lib`, `constructs`, and a test runner using `aws-cdk-lib/assertions`
    - _Requirements: 22.1_

  - [x] 1.4 Remove the obsolete moderation module and Rekognition dependency
    - Delete `backend/src/lib/moderation.ts` (the new architecture has no content-moderation step; the poll path has no `BLOCKED` status and returns `READY` directly on output presence)
    - Remove `@aws-sdk/client-rekognition` (and the now-unused `@aws-sdk/s3-request-presigner`) from `backend/package.json`; this is cleanup only and must not block other tasks
    - _Requirements: 8.3, 8.4_

- [x] 2. Implement the effect catalog and Async_Request builder (pure logic)
  - [x] 2.1 Implement `backend/src/lib/effects.ts`
    - Define `EffectCategory`, `EffectOption`, and the `EFFECTS` catalog with exactly 6 background options (Spaceship interior, Roman colosseum, Tropical background, Snowy mountain peak, Neon city street at night, Enchanted forest) and exactly 6 person options (Viking warrior, Roman emperor, Astronaut, Renaissance noble, Cyberpunk hacker, Medieval knight), each with a unique id, non-empty label, and non-empty prompt
    - Export a lookup helper that resolves an `effectId` to its prompt and rejects unknown ids
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 7.1_

  - [x]* 2.2 Write property test for the effect catalog
    - **Property 1: Effect catalog shape and total prompt mapping** — exactly 12 options (6 background + 6 person), unique ids, non-empty label and prompt for every option
    - Own file (e.g. `effects.property1.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 1 comment
    - **Validates: Requirements 5.1, 5.4, 5.6, 6.1, 6.4, 7.1**

  - [x] 2.3 Implement `backend/src/lib/request-builder.ts`
    - Define the `AsyncRequest` interface and `buildAsyncRequest` that sets `inputs` to the prompt mapped from `effectId` (via `effects.ts`), sets `images` to `[photo]`, applies booth defaults, and clamps `num_inference_steps` into `[4, 20]` and `guidance_scale` into `[1, 10]`
    - This module is imported by `ui/src/api/generation.ts`, so the browser builds requests with the same shared logic
    - _Requirements: 7.2, 7.3, 7.4_

  - [x]* 2.4 Write property test for prompt mapping and reference image
    - **Property 2: Async_Request maps the selected prompt and embeds the photo** — `inputs` equals the mapped prompt and `images` equals exactly `[photo]` (length 1, ≤ 4)
    - Own file (e.g. `request-builder.property2.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 2 comment
    - **Validates: Requirements 7.2, 7.3**

  - [x]* 2.5 Write property test for parameter clamping
    - **Property 3: Inference parameters are clamped into the booth ranges** — for any numeric inputs the result stays in `[4, 20]` / `[1, 10]`, and in-range inputs are preserved unchanged
    - Own file (e.g. `request-builder.property3.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 3 comment
    - **Validates: Requirements 7.4**

- [x] 3. Implement endpoint status/guards, authz, working-hours, and poll-decision (pure logic)
  - [x] 3.1 Implement `backend/src/lib/status-map.ts`
    - Define the shared `EndpointStatus` enum (`NOT_DEPLOYED | CREATING | IN_SERVICE | DELETING | FAILED`), a total `mapStatus` from any SageMaker status string and the not-found condition into the enum, and the `canStart`/`canStop` guard predicates
    - _Requirements: 15.1, 15.4, 16.1, 16.4, 17.1, 17.3, 18.1, 18.4_

  - [x]* 3.2 Write property test for status mapping
    - **Property 9: SageMaker status mapping is total and never over-reports availability** — every input maps inside the enum, not-found → `NOT_DEPLOYED`, only `InService` → `IN_SERVICE`
    - Own file (e.g. `status-map.property9.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 9 comment
    - **Validates: Requirements 16.1, 16.4**

  - [x]* 3.3 Write property test for endpoint action guards
    - **Property 10: Endpoint action guards depend only on current status** — start permitted iff `NOT_DEPLOYED`; stop permitted iff not `NOT_DEPLOYED`
    - Own file (e.g. `status-map.property10.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 10 comment
    - **Validates: Requirements 17.1, 17.3, 18.1, 18.4**

  - [x] 3.4 Implement `backend/src/lib/authz.ts`
    - Define the `AuthClaims` interface and `isAdmin(claims)` returning true iff `custom:profile === "ADMIN"` or `cognito:groups` includes `admin`; used client-side only to toggle the Admin tab (cosmetic — IAM is the real boundary)
    - _Requirements: 14.1, 14.2, 14.3_

  - [x]* 3.5 Write property test for admin tab visibility
    - **Property 11: Admin tab visibility is equivalent to admin identity** — show the Admin tab iff `isAdmin(claims)`; non-admin and unauthenticated claims always hide it
    - Own file (e.g. `authz.property11.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 11 comment
    - **Validates: Requirements 14.1, 14.2, 14.3**

  - [x] 3.6 Implement `backend/src/lib/working-hours.ts` (validation + serialization)
    - Define `WorkingHours`/`WorkingHoursItem`, a `validateWorkingHours` predicate requiring `endTime` strictly after `startTime`, and `toItem`/`fromItem` (de)serialization to the `pk = ENDPOINT#<name>` / `sk = DAY#<YYYY-MM-DD>` Schedule_Store shape
    - _Requirements: 20.1, 20.2_

  - [x] 3.7 Add `isWithinWindow` to `backend/src/lib/working-hours.ts`
    - Add a pure `isWithinWindow` helper that reports whether a given wall-clock time is within `[startTime, endTime)` (start inclusive, end exclusive) for a day's Working_Hours; used by both the SPA (informational) and the `Scheduler_Function`
    - _Requirements: 21.3, 21.4_

  - [x]* 3.8 Write property test for working-hours validity
    - **Property 12: Working_Hours validity requires end strictly after start** — valid iff end time strictly later than start time
    - Own file (e.g. `working-hours.property12.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 12 comment
    - **Validates: Requirements 20.2**

  - [x]* 3.9 Write property test for working-hours serialization round-trip
    - **Property 13: Working_Hours serialization round-trips** — serialize then parse yields the same day, start time, and end time
    - Own file (e.g. `working-hours.property13.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 13 comment
    - **Validates: Requirements 20.1, 20.5**

  - [x]* 3.10 Write unit test for the `isWithinWindow` boundary
    - Verify the half-open interval semantics: start time is inside the window, end time is outside, and times before start / after end are outside
    - _Requirements: 21.3, 21.4_

  - [x] 3.11 Implement `backend/src/lib/poll-decision.ts`
    - Define the pure `decidePoll({ outputPresent, failurePresent, elapsedMs, timeoutMs })` returning `READY` when output present, else `FAILED` when failure present, else `TIMEOUT` when `elapsedMs > timeoutMs` (120000), else `PENDING` — no `BLOCKED` status
    - _Requirements: 8.2, 8.4, 8.5_

  - [x]* 3.12 Write property test for the poll decision
    - **Property 4: Async poll decision honors precedence (output, failure, timeout, pending)** — across all combinations of presence flags and elapsed time the documented precedence holds; exercise the 120000 ms boundary via generators
    - Own file (e.g. `poll-decision.property4.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 4 comment
    - **Validates: Requirements 8.2, 8.4, 8.5**

- [x] 4. Implement the capture-flow state machine (pure logic)
  - [x] 4.1 Implement `ui/src/booth/machine.ts`
    - Define the machine state union, events, and a pure `transition(state, event)` reducer covering SignedOut → Start → Camera → Review → Effects → Loading → Result/Error, idle auto-reset to Start, the selection lock in Loading, and a single-active-screen derivation; terminal poll statuses are `READY | FAILED | TIMEOUT` only (no `BLOCKED`); import the shared `EffectOption`/effect ids from the effect catalog types
    - _Requirements: 1.2, 3.2, 4.1, 4.2, 5.5, 6.5, 6.6, 9.1, 9.2, 9.3, 9.4, 10.3, 10.4, 10.5_

  - [x]* 4.2 Write property test for applying poll results to Loading
    - **Property 5: Applying a poll result transitions Loading deterministically and never shows both loading and result** — `READY` → `Result` with image; `FAILED`/`TIMEOUT` → `Error` with no image; `PENDING` keeps `Loading`; exactly one screen active per state
    - Own file (e.g. `machine.property5.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 5 comment
    - **Validates: Requirements 9.2, 9.3, 9.4**

  - [x]* 4.3 Write property test for effect selection and lock-out
    - **Property 6: Effect selection records the choice, enters Loading, and locks out further selections** — `SELECT(id)` from `Effects` → `Loading` with recorded effect; subsequent `SELECT` events while `Loading` are ignored and the first selection is retained
    - Own file (e.g. `machine.property6.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 6 comment
    - **Validates: Requirements 5.5, 6.5, 6.6, 9.1**

  - [x]* 4.4 Write property test for session data clearing on return to Start
    - **Property 7: Returning to Start by any path clears all session data** — for all states and any event resulting in `Start` (New Session, idle auto-reset), the result carries neither captured photo nor transformed image
    - Own file (e.g. `machine.property7.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 7 comment
    - **Validates: Requirements 10.4, 10.5**

  - [x]* 4.5 Write property test for Review reset/continue
    - **Property 8: Continue retains the photo; Reset discards it and returns to camera** — `CONTINUE` from `Review` → `Effects` retaining the photo; `RESET` → `Camera` discarding the photo
    - Own file (e.g. `machine.property8.test.ts`); `fast-check` `{ numRuns: 100 }`; tag with the Property 8 comment
    - **Validates: Requirements 4.1, 4.2**

- [x] 5. Checkpoint - pure logic and property tests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement the client auth and credentials layer (`ui/src/auth`)
  - [x] 6.1 Implement `ui/src/auth/authService.ts`
    - Implement `getSession()` (Cognito session with implicit access/ID-token refresh; `null` when the refresh token is dead), `refreshSession()` (force a Cognito token refresh), `getCredentials(session)` (Identity-Pool STS credentials for the mapped IAM role), and `refreshCredentials()` (force a re-mint of the STS credentials), plus a `requireSignIn()` signal; expose the `profile`/`cognito:groups` claims for the cosmetic admin toggle
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 12.1, 12.2, 12.3, 12.4_

  - [x] 6.2 Implement `ui/src/auth/useAuth.ts`
    - Implement the `useAuth` hook exposing `isAuthenticated`, `isAdmin` (from the `profile` claim via `lib/authz.ts`, cosmetic), and `connectionStatus` with exactly two values — `"connected"` when a valid session and valid STS credentials are held, otherwise `"disconnected"`; never expose the username or identity
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 14.1, 14.2, 14.3_

  - [x]* 6.3 Write unit tests for connection status and identity hiding
    - Test that `connectionStatus` is `connected` iff valid session + valid STS creds (else `disconnected`), and assert that no username/identity is rendered on any screen
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

- [x] 7. Implement the browser-side AWS layer (`ui/src/api`)
  - [x] 7.1 Implement `ui/src/api/awsClients.ts` and the refresh-once-then-retry wrapper
    - Build the SDK v3 clients (S3, SageMaker Runtime, SageMaker management, DynamoDB) with a lazy refreshing credentials provider via `fromCognitoIdentityPool` that calls `authService.getSession()` then `getCredentials()`; implement `withAuthRetry` (refresh session + credentials once and retry exactly once on an auth-flavoured failure, then `requireSignIn` on a second failure) and the `isAuthError` heuristic (true for `ExpiredToken`/`ExpiredTokenException`/expired-security-token/`InvalidClientTokenId`/`NotAuthorized`/401–403; **false** for throttling/5xx/network/abort)
    - _Requirements: 11.2, 12.1, 12.2, 12.3, 12.4_

  - [x]* 7.2 Write integration test for the refresh-once-then-retry wrapper
    - Simulate auth-flavoured failures and assert exactly one `refreshSession` + `refreshCredentials` then a single retry; a second auth failure triggers `requireSignIn`; assert **no** retry on throttling/5xx/network errors
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [x] 7.3 Implement `ui/src/api/generation.ts` (`Generation_Service`)
    - **Submit:** validate `{ effectId, photo }`, build the Async_Request via `request-builder`, `PutObject` the JSON to `flux2-klein-inputs/{jobId}.json`, then `InvokeEndpointAsync` on `flux2-klein-9b-g6e2`, returning `{ jobId, submittedAt }`
    - **Poll:** `HeadObject` the output/failure keys and apply `poll-decision`; on `READY` `GetObject` the PNG bytes and build a blob/object URL (`{ status: "READY", imageUrl, aiGenerated: true }`); else map to `FAILED`/`TIMEOUT`/`PENDING`; enforce the 120 s deadline client-side and revoke the object URL on cleanup
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x]* 7.4 Write integration test for generation submit/poll
    - With mocked S3 + SageMaker Runtime, assert `PutObject` to the inputs prefix precedes `InvokeEndpointAsync` with the matching `InputLocation`; output present → `GetObject` then an object URL and `READY`; failure present → `FAILED`
    - _Requirements: 8.1, 8.3, 8.4_

  - [x] 7.5 Implement `ui/src/api/endpoints.ts` (`Endpoint_Manager`)
    - Implement `ListEndpoints` (return names; surface an invalid-credentials/region **configuration error distinct from an empty list**; empty list → "no endpoints available"), `DescribeEndpoint` (map through `status-map`, not-found/`ValidationException` → `NOT_DEPLOYED`), start (`canStart` guard → `CreateEndpoint` from the configured EndpointConfig name; reject "endpoint already exists" otherwise), and stop (`canStop` guard → `DeleteEndpoint`; "no endpoint to delete" otherwise) — all wrapped in `withAuthRetry`
    - _Requirements: 15.1, 15.4, 15.5, 16.1, 16.4, 17.1, 17.3, 18.1, 18.4_

  - [x]* 7.6 Write integration test for endpoint management
    - With mocked SageMaker management, assert list/describe/create/delete call shapes, the credential/region error vs empty-list distinction, start-on-deployed rejection, stop-on-not-deployed message, and that a mocked `AccessDeniedException` (standard-user context) surfaces as an authorization error performing no operation
    - _Requirements: 14.6, 15.4, 16.4, 17.3, 18.4_

  - [x] 7.7 Implement `ui/src/api/schedule.ts`
    - Implement schedule CRUD under the `Admin_Role`: `Query(pk = ENDPOINT#<name>)` to load the schedule, `PutItem` to upsert a day's Working_Hours (re-validating `endTime > startTime` via `working-hours`), and `DeleteItem` to remove a day — all wrapped in `withAuthRetry`
    - _Requirements: 20.1, 20.2, 20.4, 20.5_

  - [x]* 7.8 Write integration test for schedule CRUD
    - With mocked DynamoDB, assert an invalid entry is rejected before `PutItem`, a valid entry persists, delete removes the item, and reopening (`Query`) returns the persisted hours
    - _Requirements: 20.1, 20.4, 20.5_

- [x] 8. Checkpoint - browser auth + AWS layer integration tests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement the React SPA capture flow, connection indicator, and theme
  - [x] 9.1 Implement the app gate and connection indicator (`ui/src/App.tsx`)
    - Gate the capture flow behind sign-in via `useAuth`, show an auth-error message on invalid credentials, support sign-out returning to the sign-in interface, render the two-state connection indicator, gate the Admin tab on `isAdmin` (cosmetic), and ensure no username/identity is shown on any screen
    - _Requirements: 11.1, 11.3, 11.4, 13.1, 13.2, 13.3, 13.4, 14.1, 14.2_

  - [x] 9.2 Implement Start screen and camera capture (`ui/src/booth/StartScreen.tsx`, `CameraView.tsx`)
    - Render the single Start control and transition to the camera view; use `getUserMedia` for the live feed, capture a still to base64 PNG/JPEG via canvas, show/hide Take Photo for temporary-unavailable, and show a camera-error message with a retry control; drive transitions through `machine.ts`
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.4_

  - [x] 9.3 Implement review and effect selection (`ui/src/booth/ReviewScreen.tsx`, `EffectSelector.tsx`)
    - Show the captured photo with Reset/Continue, then render all 12 effect options (6 background + 6 person) from the shared catalog alongside the captured photo, record the selection, and submit via `ui/src/api/generation.ts` while locking out further selections
    - _Requirements: 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.1_

  - [x] 9.4 Implement loading and result screens (`ui/src/booth/LoadingScreen.tsx`, `ResultScreen.tsx`)
    - Show the loading state while polling, render the transformed image (never simultaneously with loading) with an AI-generated notice and a New Session control, show an error message with retry/back on failure/timeout, and discard session data on restart
    - _Requirements: 8.5, 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 9.5 Implement the portrait/touch theme (`ui/src/theme/`)
    - Implement portrait layout (height > width), ≥ 44×44 CSS-pixel touch targets, per-step minimal controls, and keyboard/mouse-free operation for the capture and effect flow
    - _Requirements: 25.1, 25.2, 25.3, 25.4_

  - [x]* 9.6 Write unit tests for capture-flow screens
    - Test Start (single control), Camera (control visibility, error + retry), Review (reset/continue, photo shown), Effects (12 options visible, captured photo shown, effect membership labels), Loading/Result (mutual exclusion, AI-generated notice, restart), and base64 capture encoding
    - _Requirements: 1.1, 2.3, 2.4, 4.3, 5.2, 5.3, 6.2, 6.3, 10.2, 10.3_

- [x] 10. Implement the admin SPA tab
  - [x] 10.1 Implement the endpoint panel (`ui/src/admin/EndpointPanel.tsx`)
    - List endpoints by name, record the selected endpoint, display + refresh status, show start (creating) status, require an explicit confirm before stop (deleting) status, and show "no endpoints available" when empty; wire to `ui/src/api/endpoints.ts`
    - _Requirements: 15.2, 15.3, 15.5, 16.2, 16.3, 17.2, 17.3, 18.2, 18.3_

  - [x] 10.2 Implement the schedule calendar (`ui/src/admin/ScheduleCalendar.tsx`)
    - Render selectable days, visually mark days with defined Working_Hours, open a per-day editor, validate `endTime > startTime` with a validation message, and persist/load via `ui/src/api/schedule.ts` so reopening shows previously defined hours
    - _Requirements: 19.1, 19.2, 19.3, 20.2, 20.3, 20.5_

  - [x] 10.3 Implement the admin tab gating (`ui/src/admin/AdminTab.tsx`)
    - Show the Admin tab only when `isAdmin` is true, hide it for standard/unauthenticated users, and wire the endpoint panel and schedule calendar into the tab
    - _Requirements: 14.1, 14.2_

  - [x]* 10.4 Write unit tests for the admin UI
    - Test admin tab visibility by role, the empty-endpoint message, the stop-confirmation requirement, the schedule validation message, and calendar day-marking
    - _Requirements: 14.1, 14.2, 15.5, 18.2, 19.2, 20.2, 20.3_

- [x] 11. Implement the Scheduler_Function (the only Lambda)
  - [x] 11.1 Implement `backend/src/scheduler/apply.ts`
    - EventBridge target that reads today's Working_Hours for the managed endpoint from `Schedule_Store` (DynamoDB), computes in/out of the `[startTime, endTime)` window in the configured timezone via `working-hours.isWithinWindow`, `DescribeEndpoint` mapped via `status-map`, and reconciles idempotently: inside-window + `NOT_DEPLOYED` → `CreateEndpoint`; outside-window + `IN_SERVICE` → `DeleteEndpoint`; otherwise no action
    - _Requirements: 21.2, 21.3, 21.4, 21.5_

  - [x]* 11.2 Write integration tests for the scheduler
    - With mocked DynamoDB + SageMaker, assert create inside-window/`NOT_DEPLOYED`, delete outside-window/`IN_SERVICE`, and a no-op when desired == actual (idempotent)
    - _Requirements: 21.2, 21.3, 21.4, 21.5_

- [x] 12. Checkpoint - SPA capture flow, admin tab, and scheduler
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement the CDK infrastructure
  - [x] 13.1 Implement `cdk/lib/auth-construct.ts`
    - Define the Cognito user pool with the `custom:profile` attribute and an `admin` group, a public SPA app client (SRP, no secret), the Identity Pool (authenticated identities only), the `Authenticated_Role` and `Admin_Role`, and the `IdentityPoolRoleAttachment` group→role mapping; attach least-privilege policies exactly as designed — `Authenticated_Role`: `InvokeEndpointAsync` on the endpoint + `s3:PutObject` on the inputs prefix + `s3:GetObject`/`s3:ListBucket` on outputs+failures (no management, no DynamoDB, no Rekognition); `Admin_Role`: adds `ListEndpoints`/`DescribeEndpoint`/`CreateEndpoint`/`DeleteEndpoint` + scoped `iam:PassRole` + DynamoDB `Query`/`PutItem`/`DeleteItem` on `Schedule_Store`; `RemovalPolicy.DESTROY`
    - _Requirements: 11.1, 14.3, 14.4, 14.5, 22.2, 22.4_

  - [x] 13.2 Implement `cdk/lib/hosting-construct.ts`
    - Define the private `UI_Bucket` (all `BlockPublicAccess` flags true, `autoDeleteObjects: true`, `RemovalPolicy.DESTROY`) and the CloudFront distribution using Origin Access Control with a bucket policy restricting access to the distribution, plus 403/404 → `/index.html` (HTTP 200) SPA fallback
    - _Requirements: 22.4, 24.1, 24.2, 24.3, 24.4_

  - [x] 13.3 Implement `cdk/lib/data-construct.ts`
    - Define the `Schedule_Store` DynamoDB table (on-demand, `RemovalPolicy.DESTROY`), the async I/O bucket (`autoDeleteObjects: true`, `RemovalPolicy.DESTROY`) or references to the existing one, and the reusable IAM policy statements consumed by the two roles in `auth-construct`
    - _Requirements: 20.1, 22.2, 22.4_

  - [x] 13.4 Implement `cdk/lib/scheduler-construct.ts`
    - Define the `Scheduler_Function` Lambda (Node 20) from `backend/src/scheduler`, its IAM role (SageMaker `ListEndpoints`/`DescribeEndpoint`/`CreateEndpoint`/`DeleteEndpoint`, scoped `iam:PassRole`, DynamoDB read on `Schedule_Store`, CloudWatch Logs), and the two EventBridge `rate(1 minute)` rules offset by ~30 s; `RemovalPolicy.DESTROY`
    - _Requirements: 21.1, 21.6, 22.2, 22.4_

  - [x] 13.5 Implement `cdk/lib/initial-users-cr.ts`
    - Define the custom resource that creates the initial Admin_User (with `custom:profile = ADMIN` and `admin` group membership) and the initial Standard_User (without admin) from the deploy-time username CfnParameters
    - _Requirements: 23.1, 23.2, 23.3, 23.4_

  - [x] 13.6 Wire the root stack and app entry (`cdk/lib/photo-booth-stack.ts`, `cdk/bin/app.ts`)
    - Instantiate all constructs, define CfnParameters for the initial admin and standard usernames and the scheduler timezone, pass shared references between constructs, ensure `RemovalPolicy.DESTROY` coverage, and assert the stack provisions **no** API Gateway and no per-request Lambda
    - _Requirements: 22.1, 22.2, 22.3, 22.4, 23.1, 23.2_

  - [x]* 13.7 Write CDK assertion/snapshot tests
    - Using `Template.fromStack`: assert `cdk synth` succeeds and the template contains the Cognito pool + client, the Identity Pool, both IAM roles with the documented least-privilege policies, the `IdentityPoolRoleAttachment`, the `Schedule_Store` table, the `UI_Bucket` + CloudFront OAC + restrictive bucket policy + SPA fallback, the `Scheduler_Function` + the two EventBridge rules, and the initial-user CfnParameters; assert **zero** `AWS::ApiGateway::*`/`AWS::ApiGatewayV2::*` resources; assert `DeletionPolicy`/`UpdateReplacePolicy: Delete` on every resource and `autoDeleteObjects` on stack-created buckets
    - _Requirements: 14.4, 14.5, 21.1, 21.6, 22.1, 22.2, 22.3, 22.4, 23.1, 23.2, 23.3, 23.4, 24.1, 24.2, 24.3, 24.4_

- [x] 14. Integration and final wiring
  - [x] 14.1 Wire the built SPA into hosting with runtime config
    - Inject the runtime config (Identity Pool ID, user pool + app client IDs, region, endpoint name, async I/O bucket, `Schedule_Store` table name) into the SPA, add a CDK `BucketDeployment` of the built `ui/` assets into `UI_Bucket` behind the distribution, and confirm the SPA loads its config and routes resolve through CloudFront
    - _Requirements: 22.2, 24.2, 24.4_

  - [x]* 14.2 Write end-to-end wiring verification tests
    - With mocked SDKs/config, verify the `ui/src/api/*` call shapes (S3 prefixes, `InvokeEndpointAsync`, SageMaker management actions, DynamoDB ops) match the IAM-permitted actions on the two roles, that there is no API Gateway or per-request Lambda, and that no resource is orphaned/unreferenced
    - _Requirements: 14.6, 22.3_

- [x] 15. Final checkpoint - full build and tests
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks (property, unit, integration, CDK assertion/snapshot) and can be skipped for a faster MVP; core implementation tasks are never optional.
- Completed pure-logic implementation tasks (`effects.ts`, `status-map.ts`, `authz.ts`, `working-hours.ts` validation/serialization, `poll-decision.ts`) and package scaffolding are marked `[x]`; their property tests, `request-builder.ts`, `isWithinWindow`, and `machine.ts` were not yet written and remain open.
- This architecture has **no per-request backend API** (no API Gateway, no per-request Lambdas) and **no content moderation**; generation, endpoint management, and schedule CRUD all run in the browser under Identity-Pool credentials authorized by IAM, and the only server-side compute is the `Scheduler_Function`.
- Property-based test sub-tasks each cover exactly one design Property (1–13, renumbered — there is no Property 14, and the former moderation property is removed), are placed immediately after the module they validate, use `fast-check` with `{ numRuns: 100 }`, carry the `// Feature: ai-photo-booth, Property {n}: ...` tag, and live in their own dedicated test file so per-property failing examples are tracked independently and tests can run in parallel.
- The pure-logic layer in `backend/src/lib/*` is shared by both the `Scheduler_Function` and the SPA (`ui/` imports it), so the same property tests protect both callers.
- The existing `flux2-klein-sagemaker.ipynb`, `prepare_weights.py`, `README.md`, and `code/` are not modified by any task.
- Checkpoints (tasks 5, 8, 12, 15) provide incremental validation breaks.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["1.4", "2.1", "3.1", "3.4", "3.6", "3.11"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.2", "3.3", "3.5", "3.7", "3.8", "3.9", "3.12", "4.1", "6.1", "13.1", "13.2", "13.3"] },
    { "id": 3, "tasks": ["2.4", "2.5", "3.10", "4.2", "4.3", "4.4", "4.5", "6.2", "7.1", "11.1"] },
    { "id": 4, "tasks": ["6.3", "7.2", "7.3", "7.5", "7.7", "9.1", "9.2", "9.5", "11.2", "13.4", "13.5"] },
    { "id": 5, "tasks": ["7.4", "7.6", "7.8", "9.3", "9.4", "10.1", "10.2"] },
    { "id": 6, "tasks": ["9.6", "10.3", "13.6"] },
    { "id": 7, "tasks": ["10.4", "13.7", "14.1"] },
    { "id": 8, "tasks": ["14.2"] }
  ]
}
```
