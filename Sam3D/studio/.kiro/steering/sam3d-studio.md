# SAM3D Studio — Project Steering

Guidance for working on the SAM3D Studio app: a web UI that turns a photo into a
3D Gaussian Splat. Point at an object → segment it in‑browser → send to a
SageMaker async endpoint → view/recolor the resulting splat.

## Architecture

- **`ui/`** — Vite + React + TypeScript SPA (Tailwind). Hosted on a private S3
  bucket behind CloudFront (OAC).
  - `src/sam.ts` — in‑browser segmentation (SlimSAM via `@huggingface/transformers`).
  - `src/viewer.ts` — Gaussian‑splat viewer (`@mkkellogg/gaussian-splats-3d`).
  - `src/material.ts` — client‑side recolor of the splat (hue rotation + strength).
  - `src/api.ts` — typed client for the API Gateway routes (Cognito ID token auth).
  - `src/components/Studio.tsx` — desktop layout.
  - `src/components/StudioMobile.tsx` — phone layout (step wizard, camera/upload).
  - `src/App.tsx` — picks desktop vs mobile via `matchMedia("(max-width: 768px)")`.
- **`cdk/`** — infra: Cognito (2 provisioned users, no self sign‑up), REST API +
  Cognito authorizer, two Lambdas (`user-api`, `admin-api`), SPA bucket +
  CloudFront, and an S3‑CORS custom resource. Does **not** create the SageMaker
  endpoint (the admin Lambda starts/stops it).
- **`deploy.sh`** — `cdk deploy` → set Cognito passwords out‑of‑band → build UI →
  sync to SPA bucket → CloudFront invalidation.
- The SageMaker container is built from the notebook (`Sam3D/sam3d-sagemaker.ipynb`).

## Data flow (and the #1 gotcha)

`/generate` calls `InvokeEndpointAsync`; the result lands in S3 and `/result`
returns a presigned URL. **The endpoint output is a JSON envelope, not a raw
`.ply`:**

```json
{ "ply_b64": "<base64 PLY>", "num_gaussians": N, "preview_gif_b64": null, "timing_s": {...} }
```

The UI must fetch that JSON, decode `ply_b64` to bytes, and pass the bytes to the
viewer. Never hand the result URL straight to the splat loader — it will try to
parse JSON as a PLY and crash with `Cannot read properties of undefined (reading
'splatCount')`. See `Studio.tsx`/`StudioMobile.tsx` `generate()`.

## In‑browser segmentation (SlimSAM) — device rules

Lessons learned, encoded in `sam.ts`:

- **Mobile → WASM only.** Mobile WebGPU is unreliable for this model:
  - fp16 on many mobile GPUs produces NaNs → salt‑and‑pepper noise masks.
  - onnxruntime‑web throws WebGPU bind‑group/Softmax validation errors on some
    Android drivers during the image encode.
  - SlimSAM is small, so WASM (CPU) runs fine; the one‑time encode is a few seconds.
- **Desktop → WebGPU** (probe `navigator.gpu.requestAdapter()` first), else WASM.
- **Use fp32, not fp16.** Precision matters more than speed here; fp16 masks are
  noisy on consumer GPUs. SlimSAM is small enough that fp32 is cheap.
- Treat a zero‑pixel mask as "no object found" and tell the user to retap, rather
  than silently doing nothing.

## Splat recolor (no endpoint change)

A Gaussian splat has no surface/UVs, so **textures/PBR materials are not possible**
(wood grain, brick pattern, fabric weave, reflectivity). Only **color** is editable:
each gaussian stores a baked SH DC color (`f_dc_0/1/2`, where `color = 0.5 + C0*f_dc`,
`C0 = 0.28209479177387814`). `material.ts` rotates each gaussian's hue and blends
toward it by a strength factor, preserving shading and region contrast. Always
re‑derive from the pristine original PLY (keep it in a ref) so changes are lossless.

If true materials are ever required, that needs a **mesh** (vertices + faces + UVs)
+ a three.js PBR material — a different representation and an endpoint change.

## Segmentation UX: SAM tap vs. manual brush

SlimSAM is **not an eraser**: every tap re-solves a single global object mask from
all points (no region subtraction, and this build has no mask-feedback input), so
"remove this part of the object" oscillates. To keep tap-refinement stable, the
session **locks one mask candidate (head)** per selection (first tap picks it;
later taps refine within it; a fresh first tap re-locks — see `maskHead` in `sam.ts`).

For deterministic control, both `Studio.tsx` and `StudioMobile.tsx` offer four
tools backed by `maskbrush.ts`:
- **Keep / Remove** — AI taps (label 1 / 0).
- **Brush / Erase** — drag to paint/rub mask pixels directly (size slider). The
  canvas uses `touch-action: none` and pointer capture so dragging doesn't scroll.
- A **brush cursor ring** matching the brush radius follows the pointer (and
  previews at center when the size slider changes), so size/position are visible.
  It's drawn in `redraw` from refs (`toolRef`/`brushRef`/`cursorRef`) so the
  `useCallback` stays dependency-free.
Keep the two UIs in sync — changes to mask tooling should land in both components.

### Mobile image pan/zoom (StudioMobile only)

Big/tall camera photos must fit and be navigable without hijacking segmentation
taps. The edit canvas sits in a fixed viewport, contained by default (whole image
visible), with a CSS `transform: translate(x,y) scale(zoom)`:
- **Zoom** ＋/－ buttons (1×–5×) and a recenter (⤢) reset.
- A **pan joystick** ("move" sphere, bottom-right) drives continuous panning via a
  rAF loop on `joyVec`; it `stopPropagation`s so it never triggers segmentation.
- No coordinate math changes are needed for taps/brush: positions come from
  `canvas.getBoundingClientRect()`, which already reflects the transform.

## Client-side video export

`recordOrbitVideo` in `viewer.ts` orbits the camera one slow turn around Y while
bobbing on X, captures the live canvas via `captureStream` + `MediaRecorder`, and
downloads it. Prefers MP4 (H.264) where supported (recent Chrome, Safari 16.4+),
falls back to WebM. No endpoint change; records whatever recolor is applied.

## Viewer notes

- The scene is unit‑normalized + centered by the handler, so set an explicit close
  camera (`initialCameraPosition ≈ [1.6, 1.1, 1.6]`, `lookAt [0,0,0]`); the viewer
  default sits far back and the model looks tiny.
- `showSplat` accepts a URL **or** `Uint8Array` (we pass decoded PLY bytes via a
  blob URL). Revoke the previous object URL on reload/dispose.

## CORS / local dev

- API Gateway `defaultCorsPreflightOptions` **and** the I/O bucket CORS list both
  `https://<cloudfront-domain>` and `http://localhost:5173`. The presigned
  PUT/GET go directly to S3, so a Vite proxy alone is insufficient — both CORS
  configs must allow the dev origin. See `allowedOrigins` in the stack.
- `npm run dev` uses the deployed backend via `ui/public/config.js` (served at
  `/config.js`), which takes precedence over `VITE_*` env vars.
- `index.html` includes `window.global ||= window;` — `amazon-cognito-identity-js`
  references `global`, which the Vite dev runtime doesn't provide (blank page
  otherwise; the production build shims it).
- Bundle assets (e.g. the sample image) locally rather than fetching cross‑origin;
  external hosts (raw.githubusercontent.com) can fail with CORS/CSP. The sample is
  `ui/public/sample.jpg`, loaded same‑origin.

## Build / deploy

- Build: `cd ui && npm run build`. The Vite build is the source of truth; a
  pre‑existing `tsc` strictness nit in `sam.ts` does not block it.
- Deploy UI only (fast path): sync `ui/dist/` to the SPA bucket with `--delete`,
  then invalidate CloudFront `"/*"`. `deploy.sh` does the full path including
  Cognito password setup.
- Verify the deployed bundle hash changed (CloudFront caches; invalidation +
  client cache‑clear/Incognito needed to pick up new assets).

## Operational reminders

- "Create 3D" only works when the SageMaker endpoint is **InService**. It starts
  **Stopped**; an admin starts it (GPU cold start is several minutes). Starting it
  incurs GPU cost — confirm before starting on someone's behalf.
- **IAM gotcha:** `sagemaker:CreateEndpoint` authorizes against BOTH the endpoint
  AND the endpoint-config resources. The admin Lambda's policy must list both
  `endpointArn` and `endpointConfigArn`, or Start fails with `AccessDeniedException`
  and no endpoint is created. (Admin Lambda errors surface as a 500 the mobile UI
  shows only faintly — check the Lambda CloudWatch logs when Start "does nothing".)
- AWS work here uses short‑lived temporary credentials; expect `ExpiredToken` and
  re‑prompt. Prefer least‑privilege; never delete/disable production safeguards.

## Key identifiers (this deployment)

- SPA bucket: `sam3dstudiostack-spabucket48e1059f-njclz2f6tied`
- CloudFront distribution: `E3HT5U6KS80UJ8`
- I/O bucket: `sagemaker-us-east-1-346399954218` (prefixes `sam3d-inputs/`,
  `sam3d-outputs/`, `sam3d-failures/`)
- Endpoint name: `sam3d-objects-g6e` (region `us-east-1`)
