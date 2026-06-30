import { useEffect, useRef, useState, useCallback } from "react";
import { SamSession, type MaskResult } from "../sam";
import { showSplat, disposeViewer, recordOrbitVideo } from "../viewer";
import { recolorPly } from "../material";
import { blankMask, stampCircle, maskHasPixels } from "../maskbrush";
import { api } from "../api";
import { signOut, type Identity } from "../auth";
import { AdminBar } from "./AdminBar";

const SAMPLE_URL = "/sample.jpg";

interface Click { x: number; y: number; label: number; }
type Stage = "capture" | "edit" | "result";
type Tool = "keep" | "remove" | "brush" | "erase";

const PROGRESS_MESSAGES = [
  "Lifting your object into 3D…",
  "Estimating shape and depth…",
  "Painting in colors and detail…",
  "Building the gaussian splat…",
  "Almost there — polishing the model…",
];

export function StudioMobile({ identity }: { identity: Identity }) {
  const samRef = useRef<SamSession | null>(null);
  const rawRef = useRef<any>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const plyRef = useRef<Uint8Array | null>(null);

  const [stage, setStage] = useState<Stage>("capture");
  const [samReady, setSamReady] = useState(false);
  const [device, setDevice] = useState("");
  const [hint, setHint] = useState("Getting the segmentation model ready…");
  const [photoUrl, setPhotoUrl] = useState<string>("");

  const [tool, setTool] = useState<Tool>("keep");
  const [clicks, setClicks] = useState<Click[]>([]);
  const [mask, setMask] = useState<MaskResult | null>(null);
  const maskRef = useRef<MaskResult | null>(null);
  const [encoding, setEncoding] = useState(false);
  const painting = useRef(false);

  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");

  const [resultTab, setResultTab] = useState<"model" | "photo">("model");
  const [hue, setHue] = useState(0);
  const [strength, setStrength] = useState(100);
  const [recording, setRecording] = useState(0); // 0 = idle, else progress 1..100
  const [brush, setBrush] = useState(5); // brush radius as % of image width

  // Keep a ref of the latest mask for pointer painting (avoids stale closures).
  useEffect(() => { maskRef.current = mask; }, [mask]);

  // Live brush cursor (position + radius preview), held in refs so redraw stays
  // dependency-free.
  const toolRef = useRef(tool); toolRef.current = tool;
  const brushRef = useRef(brush); brushRef.current = brush;
  const cursorRef = useRef<{ x: number; y: number } | null>(null);

  // Pan/zoom of the image inside a fixed viewport. Coordinates for segmentation
  // and brushing are derived from getBoundingClientRect, which already reflects
  // the CSS transform, so no extra math is needed when panned/zoomed.
  const [view, setView] = useState({ zoom: 1, x: 0, y: 0 });
  const [knob, setKnob] = useState({ x: 0, y: 0 });
  const joyVec = useRef({ x: 0, y: 0 });
  const joyActive = useRef(false);
  const joyRaf = useRef<number | null>(null);
  const joyEl = useRef<HTMLDivElement | null>(null);
  const JOY_R = 36;     // joystick radius (px)
  const PAN_SPEED = 7;  // px per frame at full deflection

  const clampPan = (v: number) => Math.max(-2500, Math.min(2500, v));

  function joyLoop() {
    const v = joyVec.current;
    if (v.x !== 0 || v.y !== 0) {
      setView((p) => ({ ...p, x: clampPan(p.x - v.x * PAN_SPEED), y: clampPan(p.y - v.y * PAN_SPEED) }));
    }
    joyRaf.current = requestAnimationFrame(joyLoop);
  }
  function updateJoy(e: React.PointerEvent) {
    const el = joyEl.current; if (!el) return;
    const r = el.getBoundingClientRect();
    let dx = e.clientX - (r.left + r.width / 2);
    let dy = e.clientY - (r.top + r.height / 2);
    const len = Math.hypot(dx, dy) || 1;
    if (len > JOY_R) { dx = (dx / len) * JOY_R; dy = (dy / len) * JOY_R; }
    setKnob({ x: dx, y: dy });
    joyVec.current = { x: dx / JOY_R, y: dy / JOY_R };
  }
  function joyDown(e: React.PointerEvent) {
    e.preventDefault(); e.stopPropagation();
    joyActive.current = true;
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    updateJoy(e);
    if (joyRaf.current == null) joyRaf.current = requestAnimationFrame(joyLoop);
  }
  function joyMove(e: React.PointerEvent) { if (joyActive.current) { e.preventDefault(); updateJoy(e); } }
  function joyUp() {
    joyActive.current = false; joyVec.current = { x: 0, y: 0 }; setKnob({ x: 0, y: 0 });
    if (joyRaf.current != null) { cancelAnimationFrame(joyRaf.current); joyRaf.current = null; }
  }
  function zoomBy(f: number) { setView((p) => ({ ...p, zoom: Math.max(1, Math.min(5, +(p.zoom * f).toFixed(2))) })); }
  function recenter() { setView({ zoom: 1, x: 0, y: 0 }); }

  // Load SAM once.
  useEffect(() => {
    const s = new SamSession();
    samRef.current = s;
    s.load()
      .then((dev) => { setDevice(dev); setSamReady(true); setHint(`Ready on ${dev.toUpperCase()}. Add a photo to begin.`); })
      .catch((e) => setHint("Couldn't load the segmentation model: " + e));
    return () => disposeViewer();
  }, []);

  // Smooth elapsed timer while generating.
  useEffect(() => {
    if (!busy) return;
    const t = window.setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => window.clearInterval(t);
  }, [busy]);

  const redraw = useCallback((m: MaskResult | null, pts: Click[]) => {
    const canvas = canvasRef.current, raw = rawRef.current;
    if (!canvas || !raw) return;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(raw.toCanvas(), 0, 0);
    if (m) {
      const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const d = img.data;
      for (let i = 0; i < m.data.length; i++) {
        if (m.data[i]) {
          // Vivid, mostly-opaque cyan highlight so the selection is obvious on
          // any background (light or dark).
          d[i * 4] = Math.round(d[i * 4] * 0.25 + 20);
          d[i * 4 + 1] = Math.round(d[i * 4 + 1] * 0.25 + 190);
          d[i * 4 + 2] = Math.round(d[i * 4 + 2] * 0.25 + 230);
        }
      }
      ctx.putImageData(img, 0, 0);
    }
    for (const c of pts) {
      ctx.beginPath();
      ctx.arc(c.x, c.y, Math.max(8, canvas.width * 0.012), 0, Math.PI * 2);
      ctx.fillStyle = c.label ? "#22c55e" : "#ef4444";
      ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = "#fff"; ctx.stroke();
    }
    const t = toolRef.current, cur = cursorRef.current;
    if (cur && (t === "brush" || t === "erase")) {
      const r = (brushRef.current / 100) * canvas.width;
      ctx.save();
      ctx.beginPath();
      ctx.arc(cur.x, cur.y, r, 0, Math.PI * 2);
      ctx.lineWidth = Math.max(2, canvas.width * 0.005);
      ctx.strokeStyle = t === "erase" ? "rgba(239,68,68,0.95)" : "rgba(255,255,255,0.95)";
      ctx.stroke();
      ctx.lineWidth = Math.max(1, canvas.width * 0.002);
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.stroke();
      ctx.restore();
    }
  }, []);

  const loadImage = useCallback(async (url: string) => {
    const sam = samRef.current; if (!sam) return;
    setError(""); setClicks([]); setMask(null); plyRef.current = null;
    setPhotoUrl(url);
    setStage("edit");
    setView({ zoom: 1, x: 0, y: 0 });
    setEncoding(true);
    setHint("Reading your photo…");
    try {
      const raw = await sam.setImage(url);
      rawRef.current = raw;
      const canvas = canvasRef.current!;
      canvas.width = raw.width; canvas.height = raw.height;
      redraw(null, []);
      setHint("Tap the object you want in 3D. Tap again to refine.");
    } catch (e: any) {
      setError("Couldn't read that image: " + (e?.message || e));
      setStage("capture");
    } finally {
      setEncoding(false);
    }
  }, [redraw]);

  const onFile = (f: File | undefined) => { if (f) loadImage(URL.createObjectURL(f)); };

  async function onCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (tool !== "keep" && tool !== "remove") return; // brush/erase handled by pointer drag
    const sam = samRef.current, canvas = canvasRef.current, raw = rawRef.current;
    if (!sam || !canvas || !raw || encoding) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const next = [...clicks, { x, y, label: tool === "keep" ? 1 : 0 }];
    setClicks(next);
    setHint("Segmenting…");
    try {
      const m = await sam.segment(canvas.width, canvas.height, next.map((c) => [c.x, c.y]), next.map((c) => c.label));
      setMask(m); redraw(m, next);
      setHint(m
        ? "Looks good? Tap “Create 3D”. Or refine with more taps."
        : "Couldn't grab an object there — try tapping its center, or add another point.");
    } catch (err: any) {
      setError("Segmentation failed: " + (err?.message || err));
    }
  }

  function undo() {
    const next = clicks.slice(0, -1);
    setClicks(next);
    if (next.length === 0) { setMask(null); redraw(null, []); return; }
    const sam = samRef.current, canvas = canvasRef.current;
    if (!sam || !canvas) return;
    sam.segment(canvas.width, canvas.height, next.map((c) => [c.x, c.y]), next.map((c) => c.label))
      .then((m) => { setMask(m); redraw(m, next); });
  }

  function clearPoints() {
    setClicks([]); setMask(null); maskRef.current = null; redraw(null, []);
    setHint("Tap the object you want in 3D.");
  }

  // ---- Manual brush / erase (deterministic mask editing) ----
  const rafPending = useRef(false);
  function scheduleBrushRedraw() {
    if (rafPending.current) return;
    rafPending.current = true;
    requestAnimationFrame(() => { rafPending.current = false; redraw(maskRef.current, clicks); });
  }
  function imgCoords(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  }
  function paintAt(x: number, y: number) {
    const canvas = canvasRef.current; if (!canvas) return;
    let m = maskRef.current;
    if (!m) { m = blankMask(canvas.width, canvas.height); maskRef.current = m; }
    stampCircle(m, x, y, (brush / 100) * canvas.width, tool === "erase" ? 0 : 255);
    scheduleBrushRedraw();
  }
  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    if (tool !== "brush" && tool !== "erase") return;
    e.preventDefault();
    painting.current = true;
    canvasRef.current?.setPointerCapture?.(e.pointerId);
    const { x, y } = imgCoords(e); cursorRef.current = { x, y }; paintAt(x, y);
  }
  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (tool !== "brush" && tool !== "erase") return;
    const { x, y } = imgCoords(e); cursorRef.current = { x, y };
    if (painting.current) { e.preventDefault(); paintAt(x, y); }
    else scheduleBrushRedraw();
  }
  function onPointerUp() {
    if (!painting.current) return;
    painting.current = false;
    const m = maskRef.current;
    if (m) {
      setMask({ data: m.data, width: m.width, height: m.height }); // commit for Create 3D
      setHint(maskHasPixels(m) ? "Looks good? Tap “Create 3D”." : "Selection is empty — brush the object or tap Keep.");
    }
  }
  function onPointerLeave() { cursorRef.current = null; scheduleBrushRedraw(); }
  function previewBrush(v: number) {
    setBrush(v); brushRef.current = v;
    const canvas = canvasRef.current;
    if (canvas) { cursorRef.current = { x: canvas.width / 2, y: canvas.height / 2 }; scheduleBrushRedraw(); }
  }

  function pngB64FromCanvas(c: HTMLCanvasElement): string { return c.toDataURL("image/png").split(",")[1]; }
  function imageB64(): string {
    const raw = rawRef.current;
    const c = document.createElement("canvas"); c.width = raw.width; c.height = raw.height;
    c.getContext("2d")!.drawImage(raw.toCanvas(), 0, 0);
    return pngB64FromCanvas(c);
  }
  function maskB64(m: MaskResult): string {
    const c = document.createElement("canvas"); c.width = m.width; c.height = m.height;
    const cx = c.getContext("2d")!; const id = cx.createImageData(m.width, m.height);
    for (let i = 0; i < m.data.length; i++) { const v = m.data[i]; id.data[i*4]=v; id.data[i*4+1]=v; id.data[i*4+2]=v; id.data[i*4+3]=255; }
    cx.putImageData(id, 0, 0);
    return pngB64FromCanvas(c);
  }

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  async function generate() {
    if (!mask) return;
    setError(""); setBusy(true); setElapsed(0); setStage("result"); setResultTab("model");
    const t0 = performance.now();
    try {
      const target = await api.uploadUrl();
      await api.putInput(target.uploadUrl, { image: imageB64(), mask: maskB64(mask), seed: 42, render_preview: false });
      const { outputKey, failureKey } = await api.generate(target.inputKey);

      let downloadUrl: string | undefined;
      const deadline = Date.now() + 20 * 60 * 1000;
      while (Date.now() < deadline) {
        await sleep(3000);
        const r = await api.result(outputKey, failureKey);
        if (r.status === "done") { downloadUrl = r.downloadUrl; break; }
        if (r.status === "error") throw new Error(r.error || "the model reported an error");
      }
      if (!downloadUrl) throw new Error("timed out after 20 minutes");

      const res = await fetch(downloadUrl);
      if (!res.ok) throw new Error(`couldn't download the result (${res.status})`);
      const env = await res.json();
      if (!env?.ply_b64) throw new Error(env?.error || "the result didn't contain a 3D model");
      const bin = atob(env.ply_b64);
      const plyBytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) plyBytes[i] = bin.charCodeAt(i);

      plyRef.current = plyBytes;
      setHue(0); setStrength(100);
      await showSplat(viewerRef.current!, plyBytes);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  // Debounced recolor from the pristine original.
  const recolorTimer = useRef<number | null>(null);
  function scheduleRecolor(nextHue: number, nextStrength: number) {
    const ply = plyRef.current;
    if (!ply) return;
    if (recolorTimer.current) window.clearTimeout(recolorTimer.current);
    recolorTimer.current = window.setTimeout(() => {
      showSplat(viewerRef.current!, recolorPly(ply, nextHue, nextStrength / 100));
    }, 120);
  }

  async function exportVideo() {
    if (!plyRef.current || recording) return;
    setResultTab("model"); setError("");
    try {
      setRecording(1);
      const { blob, ext } = await recordOrbitVideo({
        seconds: 7,
        onProgress: (p) => setRecording(Math.max(1, Math.round(p * 100))),
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = `sam3d-turntable.${ext}`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setRecording(0);
    }
  }

  function reset() {
    setStage("capture"); setClicks([]); setMask(null); plyRef.current = null;
    setError(""); setPhotoUrl(""); setHue(0); setStrength(100);
    setHint(samReady ? `Ready on ${device.toUpperCase()}. Add a photo to begin.` : "Getting ready…");
  }

  const progressMsg = PROGRESS_MESSAGES[Math.min(PROGRESS_MESSAGES.length - 1, Math.floor(elapsed / 12))];

  return (
    <div className="h-full flex flex-col bg-ink">
      {/* App bar */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-edge bg-panel/80 backdrop-blur">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-accent grid place-items-center font-bold text-white text-sm">3D</div>
          <h1 className="text-base font-semibold">SAM3D Studio</h1>
        </div>
        <button className="text-xs text-zinc-400 px-2 py-1 rounded-md border border-edge"
          onClick={() => { signOut(); location.reload(); }}>
          {identity.username} · Sign out
        </button>
      </header>

      <div className="px-4 pt-3"><AdminBar identity={identity} /></div>

      {error && (
        <div className="mx-4 mt-3 text-sm rounded-lg border border-red-500/40 bg-red-500/10 text-red-300 px-3 py-2">
          {error}
        </div>
      )}

      {/* ---- CAPTURE ---- */}
      {stage === "capture" && (
        <main className="flex-1 flex flex-col items-center justify-center gap-5 px-6 text-center">
          <div className="space-y-1">
            <h2 className="text-xl font-semibold">Turn a photo into 3D</h2>
            <p className="text-sm text-zinc-400">Snap or upload a photo, tap the object, and get a 3D model you can spin and recolor.</p>
          </div>

          <label className="w-full max-w-xs">
            <input type="file" accept="image/*" capture="environment" className="hidden"
              onChange={(e) => onFile(e.target.files?.[0])} disabled={!samReady} />
            <span className={`block w-full rounded-xl py-3.5 font-medium text-white text-center ${samReady ? "bg-accent" : "bg-zinc-700"}`}>
              📷 Take a photo
            </span>
          </label>

          <label className="w-full max-w-xs">
            <input type="file" accept="image/*" className="hidden"
              onChange={(e) => onFile(e.target.files?.[0])} disabled={!samReady} />
            <span className="block w-full rounded-xl py-3.5 font-medium text-center border border-edge bg-panel">
              🖼️ Upload from gallery
            </span>
          </label>

          <button className="text-sm text-zinc-400 underline underline-offset-4 disabled:opacity-50"
            disabled={!samReady} onClick={() => loadImage(SAMPLE_URL)}>
            Try a sample image
          </button>

          <p className="text-xs text-zinc-500 min-h-[16px]">{hint}</p>
        </main>
      )}

      {/* ---- EDIT ---- */}
      {stage === "edit" && (
        <>
          <main className="flex-1 relative overflow-hidden bg-[#0d0d12]">
            <div className="absolute inset-0 grid place-items-center p-2">
              <canvas id="imgcanvas" ref={canvasRef}
                onClick={onCanvasClick}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
                onPointerLeave={onPointerLeave}
                style={{
                  maxWidth: "100%", maxHeight: "100%", width: "auto", height: "auto",
                  transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
                  transformOrigin: "center",
                  touchAction: "none",
                }} />
            </div>

            {encoding && (
              <div className="absolute inset-0 grid place-items-center bg-ink/60">
                <div className="flex flex-col items-center gap-2 text-sm text-zinc-300">
                  <Spinner /> Reading photo…
                </div>
              </div>
            )}

            {/* Zoom + recenter controls */}
            <div className="absolute top-2 right-2 flex flex-col gap-1.5">
              <button className="w-9 h-9 rounded-full bg-panel/90 border border-edge text-lg leading-none"
                onClick={() => zoomBy(1.25)} aria-label="Zoom in">＋</button>
              <button className="w-9 h-9 rounded-full bg-panel/90 border border-edge text-lg leading-none"
                onClick={() => zoomBy(0.8)} aria-label="Zoom out">－</button>
              <button className="w-9 h-9 rounded-full bg-panel/90 border border-edge text-xs"
                onClick={recenter} aria-label="Recenter">⤢</button>
            </div>

            {/* Pan joystick — drag to move the image (doesn't trigger segmentation) */}
            <div className="absolute bottom-3 right-3 select-none" style={{ touchAction: "none" }}>
              <div ref={joyEl}
                onPointerDown={joyDown} onPointerMove={joyMove} onPointerUp={joyUp} onPointerCancel={joyUp}
                className="relative rounded-full bg-panel/80 border border-edge backdrop-blur"
                style={{ width: JOY_R * 2 + 24, height: JOY_R * 2 + 24, touchAction: "none" }}>
                <div className="absolute rounded-full bg-accent/90 border-2 border-white/70"
                  style={{
                    width: 36, height: 36,
                    left: `calc(50% - 18px + ${knob.x}px)`,
                    top: `calc(50% - 18px + ${knob.y}px)`,
                  }} />
                <span className="absolute inset-0 grid place-items-center text-[10px] text-zinc-300 pointer-events-none">move</span>
              </div>
            </div>

            {/* Hint */}
            <p className="absolute bottom-3 left-3 right-24 text-xs text-zinc-300 bg-ink/50 rounded px-2 py-1 min-h-[16px]">{hint}</p>
          </main>

          <footer className="border-t border-edge bg-panel px-4 py-3 space-y-3">
            <div className="grid grid-cols-4 gap-1.5 text-sm">
              {([
                ["keep", "✓ Keep", "Tap the object (AI selects it)."],
                ["remove", "✕ Remove", "Tap areas the AI should exclude."],
                ["brush", "🖌 Brush", "Drag to paint the selection by hand."],
                ["erase", "⌫ Erase", "Drag to rub out parts of the selection."],
              ] as [Tool, string, string][]).map(([t, label, h]) => (
                <button key={t}
                  className={`py-2 rounded-lg border border-edge ${tool === t ? "bg-accent text-white" : "bg-ink text-zinc-300"}`}
                  onClick={() => { setTool(t); setHint(h); }}>{label}</button>
              ))}
            </div>
            {(tool === "brush" || tool === "erase") && (
              <div className="flex items-center gap-3 text-sm">
                <span className="text-zinc-400 w-16">Brush</span>
                <input type="range" min={2} max={20} value={brush} className="flex-1 accent-accent"
                  onChange={(e) => previewBrush(+e.target.value)} />
              </div>
            )}
            <div className="flex items-center gap-2">
              <button className="px-3 py-2 rounded-lg border border-edge text-sm disabled:opacity-40"
                disabled={!clicks.length} onClick={undo}>Undo tap</button>
              <button className="px-3 py-2 rounded-lg border border-edge text-sm disabled:opacity-40"
                disabled={!clicks.length && !mask} onClick={clearPoints}>Clear</button>
              <button className="px-3 py-2 rounded-lg border border-edge text-sm ml-auto"
                onClick={reset}>New photo</button>
            </div>
            <button className="w-full rounded-xl py-3.5 font-semibold text-white bg-accent disabled:opacity-40"
              disabled={!mask} onClick={generate}>Create 3D model</button>
          </footer>
        </>
      )}

      {/* ---- RESULT ---- */}
      {stage === "result" && (
        <>
          <main className="flex-1 relative bg-[#0d0d12]">
            {/* Photo / 3D toggle */}
            <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 flex rounded-full overflow-hidden border border-edge bg-panel/90 backdrop-blur text-sm">
              <button className={`px-4 py-1.5 ${resultTab === "model" ? "bg-accent text-white" : "text-zinc-300"}`}
                onClick={() => setResultTab("model")}>3D model</button>
              <button className={`px-4 py-1.5 ${resultTab === "photo" ? "bg-accent text-white" : "text-zinc-300"}`}
                onClick={() => setResultTab("photo")}>Photo</button>
            </div>

            {/* Viewer (kept mounted; hidden under Photo tab) */}
            <div ref={viewerRef} className={`absolute inset-0 ${resultTab === "model" ? "" : "hidden"}`} />

            {/* Photo */}
            {resultTab === "photo" && photoUrl && (
              <div className="absolute inset-0 grid place-items-center p-3">
                <img src={photoUrl} alt="original" className="max-w-full max-h-full rounded-lg object-contain" />
              </div>
            )}

            {/* Progress overlay */}
            {busy && (
              <div className="absolute inset-0 grid place-items-center bg-ink/80 z-20">
                <div className="flex flex-col items-center gap-4 px-8 text-center">
                  <Spinner big />
                  <div className="space-y-1">
                    <p className="font-medium">{progressMsg}</p>
                    <p className="text-sm text-zinc-400">{elapsed}s elapsed{elapsed < 30 ? " · first run also warms up the model" : ""}</p>
                  </div>
                  <div className="w-56 h-1.5 rounded-full bg-edge overflow-hidden">
                    <div className="h-full bg-accent animate-pulse" style={{ width: `${Math.min(95, 10 + elapsed * 2)}%` }} />
                  </div>
                </div>
              </div>
            )}
          </main>

          <footer className="border-t border-edge bg-panel px-4 py-3 space-y-3">
            {plyRef.current && !busy && (
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-3">
                  <span className="w-16 text-zinc-400">Hue</span>
                  <input type="range" min={0} max={360} value={hue} className="flex-1 accent-accent"
                    onChange={(e) => { const v = +e.target.value; setHue(v); scheduleRecolor(v, strength); }} />
                  <span className="w-10 text-right text-zinc-500 tabular-nums">{hue}°</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="w-16 text-zinc-400">Strength</span>
                  <input type="range" min={0} max={100} value={strength} className="flex-1 accent-accent"
                    onChange={(e) => { const v = +e.target.value; setStrength(v); scheduleRecolor(hue, v); }} />
                  <span className="w-10 text-right text-zinc-500 tabular-nums">{strength}%</span>
                </div>
                <button className="text-xs text-zinc-400 underline underline-offset-4"
                  onClick={() => { setHue(0); setStrength(100); const p = plyRef.current!; showSplat(viewerRef.current!, p); }}>
                  Reset colors
                </button>
              </div>
            )}
            {plyRef.current && !busy && (
              <button className="w-full rounded-xl py-3 font-medium border border-edge disabled:opacity-50"
                disabled={!!recording} onClick={exportVideo}>
                {recording ? `Recording… ${recording}%` : "🎥 Export rotating video"}
              </button>
            )}
            <div className="flex gap-2">
              <button className="flex-1 rounded-xl py-3 font-medium border border-edge"
                onClick={() => setStage("edit")} disabled={busy}>Edit selection</button>
              <button className="flex-1 rounded-xl py-3 font-semibold text-white bg-accent"
                onClick={reset} disabled={busy}>New photo</button>
            </div>
          </footer>
        </>
      )}
    </div>
  );
}

function Spinner({ big = false }: { big?: boolean }) {
  const s = big ? "w-9 h-9 border-[3px]" : "w-5 h-5 border-2";
  return <span className={`inline-block ${s} rounded-full border-zinc-500 border-t-accent animate-spin`} />;
}
