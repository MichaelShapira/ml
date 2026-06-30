// Manual mask editing helpers. SAM gives a starting selection, but it can't be
// sculpted region-by-region (it re-solves globally on every click). A brush lets
// the user add/erase mask pixels directly for deterministic control.
import type { MaskResult } from "./sam";

export function blankMask(width: number, height: number): MaskResult {
  return { data: new Uint8Array(width * height), width, height };
}

// Stamp a filled circle into the mask. value 255 = add, 0 = erase. Mutates data.
export function stampCircle(
  m: MaskResult,
  cx: number,
  cy: number,
  radius: number,
  value: number
): void {
  const { data, width: W, height: H } = m;
  const r = Math.max(1, Math.round(radius));
  const r2 = r * r;
  const x0 = Math.max(0, Math.floor(cx - r));
  const x1 = Math.min(W - 1, Math.ceil(cx + r));
  const y0 = Math.max(0, Math.floor(cy - r));
  const y1 = Math.min(H - 1, Math.ceil(cy + r));
  for (let y = y0; y <= y1; y++) {
    const dy = y - cy;
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      if (dx * dx + dy * dy <= r2) data[y * W + x] = value;
    }
  }
}

export function maskHasPixels(m: MaskResult | null): boolean {
  if (!m) return false;
  const d = m.data;
  for (let i = 0; i < d.length; i++) if (d[i]) return true;
  return false;
}
