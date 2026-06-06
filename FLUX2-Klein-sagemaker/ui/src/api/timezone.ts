/**
 * Timezone helpers for the SPA — compute the wall-clock day/time in the
 * scheduler's configured IANA timezone, using `Intl.DateTimeFormat` only (no
 * external library). This mirrors `backend/src/scheduler/apply.ts` so the
 * browser and the scheduler agree on "now", regardless of the kiosk's own OS
 * timezone.
 */

import { getConfig } from "../config";

/** A wall-clock instant: ISO day (`YYYY-MM-DD`) and 24h time (`HH:mm`). */
export interface WallClock {
  day: string;
  time: string;
}

/** Compute the wall-clock day/time for an instant in the given IANA timezone. */
export function computeWallClock(timezone: string, at: Date = new Date()): WallClock {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(at);

  const map: Record<string, string> = {};
  for (const part of parts) {
    map[part.type] = part.value;
  }
  const hour = map.hour === "24" ? "00" : map.hour;
  return {
    day: `${map.year}-${map.month}-${map.day}`,
    time: `${hour}:${map.minute}`,
  };
}

/** The wall-clock "now" in the configured scheduler timezone. */
export function nowInScheduleTz(at: Date = new Date()): WallClock {
  return computeWallClock(getConfig().timezone, at);
}

/** Parse `HH:mm` to minutes since midnight (0–1439), or null if malformed. */
export function toMinutes(time: string): number | null {
  const m = /^([01]\d|2[0-3]):([0-5]\d)$/.exec(time);
  return m ? Number(m[1]) * 60 + Number(m[2]) : null;
}

/** Format minutes-since-midnight back to `HH:mm` (wraps within the same day). */
export function fromMinutes(total: number): string {
  const clamped = ((total % 1440) + 1440) % 1440;
  const h = Math.floor(clamped / 60);
  const m = clamped % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

/**
 * Add `minutes` to an `HH:mm` time. The result is capped at `23:59` (does not
 * roll into the next day) so a manual-start window stays within today's
 * Working_Hours item — the scheduler evaluates per-day windows.
 */
export function addMinutesCapped(time: string, minutes: number): string {
  const base = toMinutes(time);
  if (base === null) {
    return time;
  }
  return fromMinutes(Math.min(base + minutes, 23 * 60 + 59));
}

/** Return the earlier of two `HH:mm` times (malformed inputs are ignored). */
export function minTime(a: string, b: string): string {
  const ma = toMinutes(a);
  const mb = toMinutes(b);
  if (ma === null) return b;
  if (mb === null) return a;
  return ma <= mb ? a : b;
}

/** Return the later of two `HH:mm` times (malformed inputs are ignored). */
export function maxTime(a: string, b: string): string {
  const ma = toMinutes(a);
  const mb = toMinutes(b);
  if (ma === null) return b;
  if (mb === null) return a;
  return ma >= mb ? a : b;
}
