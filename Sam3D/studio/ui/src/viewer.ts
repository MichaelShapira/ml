// Interactive Gaussian-splat viewer (orbit/zoom) using @mkkellogg/gaussian-splats-3d.
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

let viewer: any = null;
let lastObjectUrl: string | null = null;
let lastRoot: HTMLElement | null = null;

// Accepts either a URL to a raw .ply, or the PLY bytes themselves. The SageMaker
// endpoint returns a JSON envelope ({ ply_b64, ... }), so callers decode the PLY
// and pass the bytes here rather than a URL to the (JSON) output object.
export async function showSplat(root: HTMLElement, ply: string | Uint8Array): Promise<void> {
  if (viewer) {
    try { await viewer.dispose(); } catch { /* ignore */ }
    viewer = null;
  }
  if (lastObjectUrl) { URL.revokeObjectURL(lastObjectUrl); lastObjectUrl = null; }
  root.innerHTML = "";
  lastRoot = root;

  let plyUrl: string;
  if (typeof ply === "string") {
    plyUrl = ply;
  } else {
    const buf = ply.buffer.slice(ply.byteOffset, ply.byteOffset + ply.byteLength) as ArrayBuffer;
    const blob = new Blob([buf], { type: "application/octet-stream" });
    plyUrl = URL.createObjectURL(blob);
    lastObjectUrl = plyUrl;
  }

  viewer = new GaussianSplats3D.Viewer({
    rootElement: root,
    sharedMemoryForWorkers: false, // avoids COOP/COEP header requirements
    dynamicScene: false,
    useBuiltInControls: true,      // OrbitControls: drag to rotate, scroll to zoom
    cameraUp: [0, 1, 0],
    // The scene is unit-normalized + centered, so a close 3/4 camera frames it
    // nicely instead of starting tiny (the viewer default sits far back).
    initialCameraPosition: [1.6, 1.1, 1.6],
    initialCameraLookAt: [0, 0, 0],
  });
  await viewer.addSplatScene(plyUrl, {
    format: GaussianSplats3D.SceneFormat.Ply,
    showLoadingUI: false,
  });
  viewer.start();
}

export function disposeViewer(): void {
  if (viewer) {
    try { viewer.dispose(); } catch { /* ignore */ }
    viewer = null;
  }
  if (lastObjectUrl) { URL.revokeObjectURL(lastObjectUrl); lastObjectUrl = null; }
}

// Pick the best-supported recording container/codec; prefer MP4 (H.264).
function pickVideoMime(): string {
  const prefs = [
    "video/mp4;codecs=avc1.42E01E",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  const sup = (window as any).MediaRecorder?.isTypeSupported;
  if (sup) for (const m of prefs) { try { if (sup(m)) return m; } catch { /* ignore */ } }
  return "video/webm";
}

export interface OrbitVideoOptions {
  seconds?: number;     // total duration
  fps?: number;         // capture frame rate
  radius?: number;      // camera distance from origin
  baseElevDeg?: number; // mean elevation above the equator
  tiltDeg?: number;     // ± X-axis bob amplitude
  onProgress?: (p: number) => void; // 0..1
}

/**
 * Records the current model: one slow full revolution around Y while bobbing up
 * and down on X. Captures the live viewer canvas. Returns the encoded blob and
 * the file extension ("mp4" where supported, else "webm").
 */
export async function recordOrbitVideo(opts: OrbitVideoOptions = {}): Promise<{ blob: Blob; ext: string }> {
  if (!viewer) throw new Error("No model loaded to record.");
  const v: any = viewer;
  const canvas: HTMLCanvasElement | null =
    v.renderer?.domElement || lastRoot?.querySelector("canvas") || null;
  const camera: any = v.camera;
  const controls: any = v.controls;
  if (!canvas || !camera) throw new Error("Viewer is not ready for capture.");
  if (typeof canvas.captureStream !== "function") throw new Error("This browser can't capture the canvas.");

  const seconds = opts.seconds ?? 7;
  const fps = opts.fps ?? 30;
  const r = opts.radius ?? 2.4;
  const baseElev = ((opts.baseElevDeg ?? 16) * Math.PI) / 180;
  const tilt = ((opts.tiltDeg ?? 22) * Math.PI) / 180;

  // Save state; freeze user controls so they don't fight our camera path.
  const savedPos = camera.position.clone();
  const prevEnabled = controls ? controls.enabled : undefined;
  if (controls) { controls.enabled = false; controls.target?.set?.(0, 0, 0); }

  const stream = canvas.captureStream(fps);
  const mimeType = pickVideoMime();
  const rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 12_000_000 });
  const chunks: BlobPart[] = [];
  rec.ondataavailable = (e: BlobEvent) => { if (e.data && e.data.size) chunks.push(e.data); };
  const stopped = new Promise<void>((res) => { rec.onstop = () => res(); });
  rec.start();

  const startT = performance.now();
  await new Promise<void>((resolve) => {
    const frame = (now: number) => {
      const p = Math.min(1, (now - startT) / (seconds * 1000));
      const a = p * Math.PI * 2;                  // full Y revolution
      const elev = baseElev + tilt * Math.sin(p * Math.PI * 2); // one X bob cycle
      const ce = Math.cos(elev), se = Math.sin(elev);
      camera.position.set(r * ce * Math.sin(a), r * se, r * ce * Math.cos(a));
      camera.up?.set?.(0, 1, 0);
      camera.lookAt(0, 0, 0);
      opts.onProgress?.(p);
      if (p < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });

  rec.stop();
  await stopped;

  // Restore the interactive camera.
  if (controls) controls.enabled = prevEnabled;
  camera.position.copy(savedPos);
  camera.lookAt(0, 0, 0);

  const ext = mimeType.includes("mp4") ? "mp4" : "webm";
  return { blob: new Blob(chunks, { type: mimeType }), ext };
}
