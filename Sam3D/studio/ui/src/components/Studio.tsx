import { useEffect, useRef, useState, useCallback } from "react";
import { SamSession, type MaskResult } from "../sam";
import { showSplat, recordOrbitVideo } from "../viewer";
import { recolorPly } from "../material";
import { blankMask, stampCircle, maskHasPixels } from "../maskbrush";
import { api } from "../api";
import type { Identity } from "../auth";
import { AdminBar } from "./AdminBar";

const SAMPLE_URL = "/sample.jpg";

interface Click { x: number; y: number; label: number; }
type Tool = "keep" | "remove" | "brush" | "erase";

export function Studio({ identity }: { identity: Identity }) {
  const samRef = useRef<SamSession | null>(null);
  const rawRef = useRef<any>(null); // RawImage
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const plyRef = useRef<Uint8Array | null>(null); // original (untinted) PLY bytes
  const maskRef = useRef<MaskResult | null>(null);
  const painting = useRef(false);

  const [device, setDevice] = useState("");
  const [status, setStatus] = useState("Loading SAM…");
  const [tool, setTool] = useState<Tool>("keep");
  const [brush, setBrush] = useState(5); // brush radius as % of image width
  const [seed, setSeed] = useState(42);
  const [clicks, setClicks] = useState<Click[]>([]);
  const [mask, setMask] = useState<MaskResult | null>(null);
  const [genStatus, setGenStatus] = useState("The 3D result appears here — drag to rotate, scroll to zoom.");
  const [busy, setBusy] = useState(false);
  const [hue, setHue] = useState(0);          // 0..360 hue rotation
  const [strength, setStrength] = useState(100); // 0..100 blend %
  const [recording, setRecording] = useState(0); // 0 = idle, else progress 1..100

  useEffect(() => { maskRef.current = mask; }, [mask]);

  // Live brush cursor (position + radius preview). Kept in refs so redraw's
  // useCallback stays dependency-free.
  const toolRef = useRef(tool); toolRef.current = tool;
  const brushRef = useRef(brush); brushRef.current = brush;
  const cursorRef = useRef<{ x: number; y: number } | null>(null);

  // Load SAM once.
  useEffect(() => {
    const s = new SamSession();
    samRef.current = s;
    s.load()
      .then((dev) => { setDevice(dev); setStatus(`SAM ready on ${dev.toUpperCase()}. Load an image and click the object.`); })
      .catch((e) => setStatus("Failed to load SAM: " + e));
  }, []);

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
          d[i * 4] = Math.round(d[i * 4] * 0.25 + 20);
          d[i * 4 + 1] = Math.round(d[i * 4 + 1] * 0.25 + 190);
          d[i * 4 + 2] = Math.round(d[i * 4 + 2] * 0.25 + 230);
        }
      }
      ctx.putImageData(img, 0, 0);
    }
    for (const c of pts) {
      ctx.beginPath();
      ctx.arc(c.x, c.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = c.label ? "#00e000" : "#ff2828";
      ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke();
    }
    drawBrushCursor(ctx, canvas);
  }, []);

  // Draw the brush outline so the user can see its size/position.
  function drawBrushCursor(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) {
    const t = toolRef.current, cur = cursorRef.current;
    if (!cur || (t !== "brush" && t !== "erase")) return;
    const r = (brushRef.current / 100) * canvas.width;
    ctx.save();
    ctx.beginPath();
    ctx.arc(cur.x, cur.y, r, 0, Math.PI * 2);
    ctx.lineWidth = Math.max(2, canvas.width * 0.004);
    ctx.strokeStyle = t === "erase" ? "rgba(239,68,68,0.95)" : "rgba(255,255,255,0.95)";
    ctx.stroke();
    ctx.lineWidth = Math.max(1, canvas.width * 0.0015);
    ctx.strokeStyle = "rgba(0,0,0,0.7)"; // contrast ring
    ctx.stroke();
    ctx.restore();
  }

  const loadImage = useCallback(async (url: string) => {
    const sam = samRef.current; if (!sam) return;
    setStatus("Encoding image (one-time)…");
    setClicks([]); setMask(null);
    const raw = await sam.setImage(url);
    rawRef.current = raw;
    const canvas = canvasRef.current!;
    canvas.width = raw.width; canvas.height = raw.height;
    redraw(null, []);
    setStatus("Image ready. Click the object; click again to refine.");
  }, [redraw]);

  async function onCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (tool !== "keep" && tool !== "remove") return; // brush/erase handled by pointer drag
    const sam = samRef.current, canvas = canvasRef.current, raw = rawRef.current;
    if (!sam || !canvas || !raw) { setStatus("Load an image first."); return; }
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const next = [...clicks, { x, y, label: tool === "keep" ? 1 : 0 }];
    setClicks(next);
    setStatus("Segmenting…");
    try {
      const m = await sam.segment(
        canvas.width, canvas.height,
        next.map((c) => [c.x, c.y]),
        next.map((c) => c.label)
      );
      setMask(m); redraw(m, next);
      setStatus(`Mask ready (${next.length} point${next.length > 1 ? "s" : ""}). Refine, or Generate 3D.`);
    } catch (err: any) {
      setStatus("Segmentation failed: " + err);
    }
  }

  // ---- Manual brush / erase (deterministic mask editing) ----
  const rafPending = useRef(false);
  function scheduleBrushRedraw() {
    if (rafPending.current) return;
    rafPending.current = true;
    requestAnimationFrame(() => { rafPending.current = false; redraw(maskRef.current, clicks); });
  }
  function paintAt(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    let m = maskRef.current;
    if (!m) { m = blankMask(canvas.width, canvas.height); maskRef.current = m; }
    stampCircle(m, x, y, (brush / 100) * canvas.width, tool === "erase" ? 0 : 255);
    scheduleBrushRedraw();
  }
  function cursorFromEvent(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    cursorRef.current = {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  }
  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    if (tool !== "brush" && tool !== "erase") return;
    e.preventDefault();
    painting.current = true;
    canvasRef.current?.setPointerCapture?.(e.pointerId);
    cursorFromEvent(e);
    paintAt(e);
  }
  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (tool !== "brush" && tool !== "erase") return;
    cursorFromEvent(e);
    if (painting.current) { e.preventDefault(); paintAt(e); }
    else scheduleBrushRedraw(); // hover: just move the cursor ring
  }
  function onPointerUp() {
    if (!painting.current) return;
    painting.current = false;
    const m = maskRef.current;
    if (m) {
      setMask({ data: m.data, width: m.width, height: m.height });
      setStatus(maskHasPixels(m) ? "Mask edited. Refine, or Generate 3D." : "Selection is empty.");
    }
  }
  function onPointerLeave() {
    cursorRef.current = null;
    scheduleBrushRedraw();
  }
  // Preview the brush size at canvas center when the slider changes.
  function previewBrush(v: number) {
    setBrush(v); brushRef.current = v;
    const canvas = canvasRef.current;
    if (canvas) { cursorRef.current = { x: canvas.width / 2, y: canvas.height / 2 }; scheduleBrushRedraw(); }
  }

  function clearPoints() {
    setClicks([]); setMask(null); maskRef.current = null; redraw(null, []);
    setStatus("Cleared. Click the object.");
  }

  function pngB64FromCanvas(c: HTMLCanvasElement): string {
    return c.toDataURL("image/png").split(",")[1];
  }
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

  async function exportVideo() {
    if (!plyRef.current || recording) return;
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
      setGenStatus("Video export failed: " + (e?.message || e));
    } finally {
      setRecording(0);
    }
  }

  // Re-derive the recolored splat from the pristine original. Debounced so
  // dragging a slider doesn't reload on every pixel.
  const recolorTimer = useRef<number | null>(null);
  function scheduleRecolor(nextHue: number, nextStrength: number) {
    const ply = plyRef.current;
    if (!ply || busy) return;
    if (recolorTimer.current) window.clearTimeout(recolorTimer.current);
    recolorTimer.current = window.setTimeout(async () => {
      await showSplat(viewerRef.current!, recolorPly(ply, nextHue, nextStrength / 100));
    }, 120);
  }

  async function generate() {
    if (!mask) return;
    setBusy(true);
    const t0 = performance.now();
    setGenStatus("Uploading & submitting to the endpoint…");
    try {
      const target = await api.uploadUrl();
      await api.putInput(target.uploadUrl, { image: imageB64(), mask: maskB64(mask), seed, render_preview: false });
      const { outputKey, failureKey } = await api.generate(target.inputKey);

      let downloadUrl: string | undefined;
      const deadline = Date.now() + 20 * 60 * 1000;
      while (Date.now() < deadline) {
        await sleep(3000);
        const r = await api.result(outputKey, failureKey);
        if (r.status === "done") { downloadUrl = r.downloadUrl; break; }
        if (r.status === "error") throw new Error(r.error || "endpoint error");
        setGenStatus(`Working on the GPU… ${Math.round((performance.now() - t0) / 1000)}s (first run also loads the model)`);
      }
      if (!downloadUrl) throw new Error("timed out after 20 min");

      // The endpoint output is a JSON envelope ({ ply_b64, num_gaussians, ... }),
      // not a raw .ply. Fetch it and decode the PLY bytes for the splat viewer.
      setGenStatus("Downloading 3D result…");
      const res = await fetch(downloadUrl);
      if (!res.ok) throw new Error(`download failed: ${res.status}`);
      const env = await res.json();
      const plyB64: string | undefined = env?.ply_b64;
      if (!plyB64) throw new Error(env?.error || "result did not contain a PLY (ply_b64 missing)");
      const bin = atob(plyB64);
      const plyBytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) plyBytes[i] = bin.charCodeAt(i);

      const total = Math.round((performance.now() - t0) / 1000);
      const n = env?.num_gaussians ? ` (${env.num_gaussians.toLocaleString()} gaussians)` : "";
      setGenStatus(`Done in ${total}s${n} — drag to rotate, scroll to zoom.`);
      plyRef.current = plyBytes;
      await showSplat(viewerRef.current!, recolorPly(plyBytes, hue, strength / 100));
    } catch (e: any) {
      setGenStatus("Generation failed: " + (e?.message || e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex-1 p-4 grid gap-4 lg:grid-cols-2 overflow-auto">
      {/* Left: image + controls */}
      <div className="bg-panel border border-edge rounded-xl p-3 space-y-3">
        <div className="flex items-center gap-2 flex-wrap">
          <input type="file" accept="image/*"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) loadImage(URL.createObjectURL(f)); }} />
          <button className="px-3 py-1.5 rounded-md border border-edge hover:bg-ink"
            onClick={() => loadImage(SAMPLE_URL)}>Use sample</button>
        </div>
        <canvas id="imgcanvas" ref={canvasRef}
          onClick={onCanvasClick}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onPointerLeave={onPointerLeave}
          style={{ touchAction: "none", cursor: (tool === "brush" || tool === "erase") ? "none" : "crosshair" }} />
        <div className="flex items-center gap-1.5 text-sm flex-wrap">
          {([
            ["keep", "✓ Keep"],
            ["remove", "✕ Remove"],
            ["brush", "🖌 Brush"],
            ["erase", "⌫ Erase"],
          ] as [Tool, string][]).map(([t, label]) => (
            <button key={t}
              className={`px-3 py-1.5 rounded-md border border-edge ${tool === t ? "bg-accent text-white" : "hover:bg-ink"}`}
              onClick={() => setTool(t)}>{label}</button>
          ))}
          {(tool === "brush" || tool === "erase") && (
            <label className="flex items-center gap-2 ml-1">
              <span className="text-zinc-400">Size</span>
              <input type="range" min={2} max={20} value={brush} className="accent-accent"
                onChange={(e) => previewBrush(parseInt(e.target.value, 10))} />
            </label>
          )}
        </div>
        <p className="text-xs text-zinc-500">
          Keep/Remove use AI selection. Brush/Erase paint the mask by hand for exact control.
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <button className="px-3 py-1.5 rounded-md border border-edge hover:bg-ink" onClick={clearPoints}>Clear points</button>
          <button className="px-4 py-1.5 rounded-md bg-accent text-white font-medium disabled:opacity-50"
            disabled={!mask || busy} onClick={generate}>Generate 3D</button>
          <label className="flex items-center gap-1.5 text-sm">Seed
            <input type="number" className="w-20 bg-ink border border-edge rounded-md px-2 py-1"
              value={seed} min={0} max={9999} onChange={(e) => setSeed(parseInt(e.target.value || "42", 10))} />
          </label>
        </div>
        <p className="text-xs text-zinc-400 min-h-[16px] whitespace-pre-wrap">{status}{device ? "" : ""}</p>
      </div>

      {/* Right: viewer + admin */}
      <div className="bg-panel border border-edge rounded-xl p-3 space-y-3">
        <AdminBar identity={identity} />
        <div className="space-y-2 text-sm">
          <div className="flex items-center gap-3">
            <label htmlFor="hue" className="text-zinc-400 w-16">Hue</label>
            <input id="hue" type="range" min={0} max={360} value={hue} className="flex-1 accent-accent"
              onChange={(e) => { const v = parseInt(e.target.value, 10); setHue(v); scheduleRecolor(v, strength); }} />
            <span className="text-zinc-500 w-10 text-right tabular-nums">{hue}°</span>
          </div>
          <div className="flex items-center gap-3">
            <label htmlFor="strength" className="text-zinc-400 w-16">Strength</label>
            <input id="strength" type="range" min={0} max={100} value={strength} className="flex-1 accent-accent"
              onChange={(e) => { const v = parseInt(e.target.value, 10); setStrength(v); scheduleRecolor(hue, v); }} />
            <span className="text-zinc-500 w-10 text-right tabular-nums">{strength}%</span>
          </div>
        </div>
        <div ref={viewerRef} className="w-full h-[460px] bg-[#0d0d12] rounded-lg overflow-hidden relative" />
        <button className="px-3 py-1.5 rounded-md border border-edge hover:bg-ink text-sm disabled:opacity-50"
          disabled={!!recording} onClick={exportVideo}>
          {recording ? `Recording… ${recording}%` : "🎥 Export rotating video"}
        </button>
        <p className="text-xs text-zinc-400 whitespace-pre-wrap">{genStatus}</p>
      </div>
    </div>
  );
}
