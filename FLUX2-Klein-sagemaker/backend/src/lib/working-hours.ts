/**
 * Working_Hours pure logic for the AI Photo Booth Schedule_Store.
 *
 * This module is intentionally free of any AWS SDK calls or live DynamoDB I/O.
 * It provides:
 *   - The `WorkingHoursItem` DynamoDB item shape.
 *   - A `validateWorkingHours` predicate requiring `endTime` strictly after `startTime`.
 *   - `toItem` / `fromItem` (de)serialization to the Schedule_Store key shape
 *     (`pk = ENDPOINT#<name>`, `sk = DAY#<YYYY-MM-DD>`).
 *
 * Requirements: 19.1 (persist/read Working_Hours), 19.2 (reject end <= start).
 * Design Properties: 13 (validity requires end strictly after start),
 * 14 (serialization round-trips day/startTime/endTime).
 */

/** Prefix applied to the partition key for an endpoint's schedule items. */
export const PK_PREFIX = "ENDPOINT#";
/** Prefix applied to the sort key for a single day's Working_Hours. */
export const SK_PREFIX = "DAY#";

/**
 * The logical Working_Hours for a single day, independent of the persisted
 * DynamoDB key shape. This is the value the API layer works with.
 */
export interface WorkingHours {
  /** Endpoint these hours apply to, e.g. `flux2-klein-9b-g6e2`. */
  endpointName: string;
  /** ISO date string, `YYYY-MM-DD`. */
  day: string;
  /** Start time, `HH:mm` (24h). */
  startTime: string;
  /** End time, `HH:mm` (24h); must be strictly after `startTime`. */
  endTime: string;
  /** Admin username that last wrote the entry. */
  updatedBy: string;
  /** ISO timestamp of the last write. */
  updatedAt: string;
}

/**
 * The Schedule_Store DynamoDB item shape. `pk`/`sk` carry the single-table
 * key design; the remaining attributes mirror the logical Working_Hours.
 */
export interface WorkingHoursItem {
  /** `ENDPOINT#<name>` */
  pk: string;
  /** `DAY#<YYYY-MM-DD>` */
  sk: string;
  /** `YYYY-MM-DD` */
  day: string;
  /** `HH:mm` */
  startTime: string;
  /** `HH:mm`, strictly after `startTime` */
  endTime: string;
  /** admin username */
  updatedBy: string;
  /** ISO timestamp */
  updatedAt: string;
}

/** Matches a 24-hour `HH:mm` time, `00:00`–`23:59`. */
const TIME_RE = /^([01]\d|2[0-3]):([0-5]\d)$/;

/**
 * Parse an `HH:mm` time into minutes since midnight, or `null` when the
 * string is not a valid 24-hour time.
 */
function toMinutes(time: string): number | null {
  const match = TIME_RE.exec(time);
  if (match === null) {
    return null;
  }
  const hours = Number(match[1]);
  const minutes = Number(match[2]);
  return hours * 60 + minutes;
}

/**
 * Working_Hours are valid iff both `startTime` and `endTime` are well-formed
 * 24-hour `HH:mm` times and `endTime` is strictly later than `startTime`
 * (Requirement 19.2, Property 13).
 */
export function validateWorkingHours(startTime: string, endTime: string): boolean {
  const start = toMinutes(startTime);
  const end = toMinutes(endTime);
  if (start === null || end === null) {
    return false;
  }
  return end > start;
}

/**
 * Report whether the wall-clock time `now` falls within the half-open
 * Working_Hours interval `[startTime, endTime)` — start inclusive, end
 * exclusive. All three arguments are 24-hour `HH:mm` strings.
 *
 * This is **fail-closed**: it returns `false` when any time string is
 * malformed, or when the window itself is invalid (i.e. `endTime` is not
 * strictly after `startTime`, per `validateWorkingHours`). The Scheduler_Function
 * uses this to gate `CreateEndpoint`/`DeleteEndpoint`, so bad or missing data
 * must never be treated as "inside the window" and start the endpoint.
 *
 * Pure and side-effect free; reused by the Scheduler_Function and the SPA.
 *
 * Requirements: 21.3 (inside `[start, end)` → desired running),
 * 21.4 (outside `[start, end)` → desired stopped).
 */
export function isWithinWindow(now: string, startTime: string, endTime: string): boolean {
  if (!validateWorkingHours(startTime, endTime)) {
    return false;
  }
  const current = toMinutes(now);
  if (current === null) {
    return false;
  }
  // start/end are guaranteed well-formed by validateWorkingHours above.
  const start = toMinutes(startTime) as number;
  const end = toMinutes(endTime) as number;
  return current >= start && current < end;
}

/** Build the partition key for an endpoint's schedule items. */
export function makePk(endpointName: string): string {
  return `${PK_PREFIX}${endpointName}`;
}

/** Build the sort key for a single day's Working_Hours. */
export function makeSk(day: string): string {
  return `${SK_PREFIX}${day}`;
}

/**
 * Serialize logical Working_Hours into the Schedule_Store item shape,
 * deriving `pk`/`sk` from the endpoint name and day (Requirement 19.1).
 */
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

/**
 * Parse a Schedule_Store item back into logical Working_Hours, recovering the
 * endpoint name from the `pk` prefix (Requirement 19.1, Property 14).
 */
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
