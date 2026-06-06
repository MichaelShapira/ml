/**
 * Working_Hours validation + Schedule_Store (de)serialization for the SPA.
 *
 * Mirrors `backend/src/lib/working-hours.ts` (the parts the browser needs).
 * Single-table key design: pk = `ENDPOINT#<name>`, sk = `DAY#<YYYY-MM-DD>`.
 */

export const PK_PREFIX = "ENDPOINT#";
export const SK_PREFIX = "DAY#";

/** Logical Working_Hours for a single day. */
export interface WorkingHours {
  endpointName: string;
  /** ISO date, `YYYY-MM-DD`. */
  day: string;
  /** Start time, `HH:mm` (24h). */
  startTime: string;
  /** End time, `HH:mm` (24h); must be strictly after `startTime`. */
  endTime: string;
  updatedBy: string;
  updatedAt: string;
}

/** The Schedule_Store DynamoDB item shape (Document-client friendly). */
export interface WorkingHoursItem {
  pk: string;
  sk: string;
  day: string;
  startTime: string;
  endTime: string;
  updatedBy: string;
  updatedAt: string;
}

const TIME_RE = /^([01]\d|2[0-3]):([0-5]\d)$/;

/** Parse `HH:mm` into minutes since midnight, or `null` when malformed. */
export function toMinutes(time: string): number | null {
  const match = TIME_RE.exec(time);
  if (match === null) {
    return null;
  }
  return Number(match[1]) * 60 + Number(match[2]);
}

/** Valid iff both times are well-formed and end is strictly after start. */
export function validateWorkingHours(startTime: string, endTime: string): boolean {
  const start = toMinutes(startTime);
  const end = toMinutes(endTime);
  if (start === null || end === null) {
    return false;
  }
  return end > start;
}

export function makePk(endpointName: string): string {
  return `${PK_PREFIX}${endpointName}`;
}
export function makeSk(day: string): string {
  return `${SK_PREFIX}${day}`;
}

export function toItem(hours: WorkingHours): WorkingHoursItem {
  return {
    pk: makePk(hours.endpointName),
    sk: makeSk(hours.day),
    day: hours.day,
    startTime: hours.startTime,
    endTime: hours.endTime,
    updatedBy: hours.updatedBy,
    updatedAt: hours.updatedAt,
  };
}

export function fromItem(item: WorkingHoursItem): WorkingHours {
  return {
    endpointName: item.pk.startsWith(PK_PREFIX)
      ? item.pk.slice(PK_PREFIX.length)
      : item.pk,
    day: item.day,
    startTime: item.startTime,
    endTime: item.endTime,
    updatedBy: item.updatedBy,
    updatedAt: item.updatedAt,
  };
}
