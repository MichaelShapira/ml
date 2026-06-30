// Client-side recolor for a 3D Gaussian Splat PLY.
//
// A splat has no surface/UVs, so true textures/PBR materials don't apply. But
// every gaussian carries its own baked color (the SH DC term f_dc_0/1/2, where
// color = 0.5 + C0 * f_dc), so we CAN recolor. We rotate each gaussian's hue and
// blend toward it by a strength factor — this preserves the object's shading and
// the contrast between regions, reading as "new colors" rather than a flat wash.

const C0 = 0.28209479177387814;

interface PlyLayout {
  headerEnd: number;             // byte offset just past "end_header\n"
  count: number;                 // number of vertices
  stride: number;                // bytes per vertex
  fdc: [number, number, number]; // byte offsets of f_dc_0/1/2 within a vertex
}

// Parse a binary-little-endian PLY header to locate the f_dc color fields.
function parseLayout(bytes: Uint8Array): PlyLayout | null {
  const text = new TextDecoder("latin1").decode(bytes.subarray(0, Math.min(bytes.length, 8192)));
  const marker = "end_header\n";
  const idx = text.indexOf(marker);
  if (idx < 0 || !text.startsWith("ply")) return null;
  if (!/format\s+binary_little_endian/.test(text)) return null;
  const headerEnd = idx + marker.length;

  const headerLines = text.slice(0, idx).split("\n");
  let count = 0;
  const props: { name: string; size: number }[] = [];
  const sizeOf: Record<string, number> = {
    char: 1, uchar: 1, int8: 1, uint8: 1,
    short: 2, ushort: 2, int16: 2, uint16: 2,
    int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4,
    double: 8, float64: 8,
  };
  for (const line of headerLines) {
    const l = line.trim();
    if (l.startsWith("element vertex")) count = parseInt(l.split(/\s+/)[2], 10);
    else if (l.startsWith("property")) {
      const parts = l.split(/\s+/); // property <type> <name>
      const type = parts[1];
      const name = parts[parts.length - 1];
      props.push({ name, size: sizeOf[type] ?? 4 });
    }
  }
  let offset = 0;
  const offsetOf: Record<string, number> = {};
  for (const p of props) { offsetOf[p.name] = offset; offset += p.size; }
  if (
    offsetOf.f_dc_0 === undefined ||
    offsetOf.f_dc_1 === undefined ||
    offsetOf.f_dc_2 === undefined
  ) return null;
  return { headerEnd, count, stride: offset, fdc: [offsetOf.f_dc_0, offsetOf.f_dc_1, offsetOf.f_dc_2] };
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

// RGB (0..1) <-> HSL (h in 0..360, s/l in 0..1).
function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0, s = 0;
  const d = max - min;
  if (d > 1e-6) {
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  return [h, s, l];
}
function hue2rgb(p: number, q: number, t: number): number {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1 / 6) return p + (q - p) * 6 * t;
  if (t < 1 / 2) return q;
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
  return p;
}
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  h /= 360;
  if (s < 1e-6) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [hue2rgb(p, q, h + 1 / 3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1 / 3)];
}

/**
 * Returns a NEW PLY buffer with hues rotated by `hueShiftDeg` (0..360) and the
 * result blended with the original by `strength` (0..1). hueShiftDeg 0 or
 * strength 0 returns an unchanged copy. The source buffer is never mutated, so
 * callers can re-derive any look from the pristine original.
 */
export function recolorPly(source: Uint8Array, hueShiftDeg: number, strength: number): Uint8Array {
  const out = source.slice();
  if (strength <= 0 || hueShiftDeg % 360 === 0) return out;

  const layout = parseLayout(out);
  if (!layout) return out; // unknown layout — don't risk corrupting it

  const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);
  const { headerEnd, count, stride, fdc } = layout;
  const k = clamp01(strength);

  for (let i = 0; i < count; i++) {
    const base = headerEnd + i * stride;
    const r = clamp01(0.5 + C0 * dv.getFloat32(base + fdc[0], true));
    const g = clamp01(0.5 + C0 * dv.getFloat32(base + fdc[1], true));
    const b = clamp01(0.5 + C0 * dv.getFloat32(base + fdc[2], true));

    const [h, s, l] = rgbToHsl(r, g, b);
    const [nr, ng, nb] = hslToRgb((h + hueShiftDeg) % 360, s, l);

    const fr = r + (nr - r) * k;
    const fg = g + (ng - g) * k;
    const fb = b + (nb - b) * k;

    dv.setFloat32(base + fdc[0], (clamp01(fr) - 0.5) / C0, true);
    dv.setFloat32(base + fdc[1], (clamp01(fg) - 0.5) / C0, true);
    dv.setFloat32(base + fdc[2], (clamp01(fb) - 0.5) / C0, true);
  }
  return out;
}
