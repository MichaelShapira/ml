/**
 * ScheduleCalendar — admin working-hours calendar (Requirements 19, 20).
 *
 * Renders a navigable month grid (month name + day names + prev/next month),
 * visually marks days that have defined Working_Hours (Req 19.2), opens a
 * per-day editor (Req 19.3), validates `endTime > startTime` with a message
 * (Req 20.2), and persists/loads via `api/schedule.ts` so reopening shows
 * previously defined hours (Req 20.5).
 *
 * Times are interpreted in the scheduler's configured timezone, so "Today" is
 * computed in that timezone — not the kiosk browser's local timezone.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { KioskScreen, TouchButton, PrimaryButton } from "../theme";
import {
  listWorkingHours,
  putWorkingHours,
  deleteWorkingHours,
} from "../api/schedule";
import { validateWorkingHours, type WorkingHours } from "../api/workingHours";
import { nowInScheduleTz } from "../api/timezone";
import {
  getDailyCostByService,
  getMonthToDateCost,
  formatCost,
  type DailyCost,
  type MonthCost,
} from "../api/cost";

const WEEKDAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTH_LABELS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

/** Format y/m(0-based)/d as `YYYY-MM-DD`. */
function isoDay(year: number, month0: number, day: number): string {
  return `${year}-${String(month0 + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

interface MonthCell {
  /** `YYYY-MM-DD` for a real day, or null for a leading blank pad cell. */
  day: string | null;
  /** Day-of-month number for a real day. */
  dayNum?: number;
}

/** Build the grid cells for a month: leading blanks + each day. */
function monthCells(year: number, month0: number): MonthCell[] {
  const firstWeekday = new Date(year, month0, 1).getDay(); // 0=Sun
  const daysInMonth = new Date(year, month0 + 1, 0).getDate();
  const cells: MonthCell[] = [];
  for (let i = 0; i < firstWeekday; i++) {
    cells.push({ day: null });
  }
  for (let d = 1; d <= daysInMonth; d++) {
    cells.push({ day: isoDay(year, month0, d), dayNum: d });
  }
  return cells;
}

export function ScheduleCalendar() {
  const [hoursByDay, setHoursByDay] = useState<Map<string, WorkingHours>>(new Map());
  const [selectedDay, setSelectedDay] = useState<string | null>(null);
  const [startTime, setStartTime] = useState("09:00");
  const [endTime, setEndTime] = useState("17:00");
  const [validationError, setValidationError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // --- Cost panel state -----------------------------------------------------
  // Month-to-date total for the booth's AiPhoto-tagged resources, and the
  // selected day's per-service breakdown. Both come from Cost Explorer.
  const [monthCost, setMonthCost] = useState<MonthCost | null>(null);
  const [monthCostError, setMonthCostError] = useState<string | null>(null);
  const [dailyCost, setDailyCost] = useState<DailyCost | null>(null);
  const [dailyCostLoading, setDailyCostLoading] = useState(false);
  const [dailyCostError, setDailyCostError] = useState<string | null>(null);

  // "Today" in the scheduler's timezone (e.g. Asia/Jerusalem), so highlighting
  // matches what the scheduler considers the current day.
  const todayIso = useMemo(() => nowInScheduleTz().day, []);

  // The visible month, initialised to the current month (scheduler tz).
  const [view, setView] = useState(() => {
    const [y, m] = todayIso.split("-").map(Number);
    return { year: y, month0: m - 1 };
  });

  const cells = useMemo(() => monthCells(view.year, view.month0), [view]);

  const reload = useCallback(async () => {
    const all = await listWorkingHours();
    const map = new Map<string, WorkingHours>();
    for (const wh of all) {
      map.set(wh.day, wh);
    }
    setHoursByDay(map);
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  // Load month-to-date cost once (today's month). Best-effort: a denial or an
  // empty Cost Explorer (tag not yet active / no usage) surfaces as a message,
  // not a crash.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const mc = await getMonthToDateCost(todayIso);
        if (!cancelled) setMonthCost(mc);
      } catch {
        if (!cancelled)
          setMonthCostError("Cost data is unavailable right now.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [todayIso]);

  // Load the selected day's per-service cost breakdown whenever the selection
  // changes (and clear it when nothing is selected).
  useEffect(() => {
    if (!selectedDay) {
      setDailyCost(null);
      setDailyCostError(null);
      return;
    }
    let cancelled = false;
    setDailyCost(null);
    setDailyCostError(null);
    setDailyCostLoading(true);
    (async () => {
      try {
        const dc = await getDailyCostByService(selectedDay);
        if (!cancelled) setDailyCost(dc);
      } catch {
        if (!cancelled) setDailyCostError("Cost data is unavailable right now.");
      } finally {
        if (!cancelled) setDailyCostLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedDay]);

  const goPrevMonth = useCallback(() => {
    setSelectedDay(null);
    setView((v) =>
      v.month0 === 0
        ? { year: v.year - 1, month0: 11 }
        : { year: v.year, month0: v.month0 - 1 },
    );
  }, []);

  const goNextMonth = useCallback(() => {
    setSelectedDay(null);
    setView((v) =>
      v.month0 === 11
        ? { year: v.year + 1, month0: 0 }
        : { year: v.year, month0: v.month0 + 1 },
    );
  }, []);

  const openEditor = useCallback(
    (day: string) => {
      setSelectedDay(day);
      setValidationError(null);
      const existing = hoursByDay.get(day);
      setStartTime(existing?.startTime ?? "09:00");
      setEndTime(existing?.endTime ?? "17:00");
    },
    [hoursByDay],
  );

  const onSave = useCallback(async () => {
    if (!selectedDay) return;
    if (!validateWorkingHours(startTime, endTime)) {
      setValidationError("End time must be later than start time.");
      return;
    }
    setBusy(true);
    try {
      await putWorkingHours({ day: selectedDay, startTime, endTime });
      await reload();
      setSelectedDay(null);
    } finally {
      setBusy(false);
    }
  }, [selectedDay, startTime, endTime, reload]);

  const onRemove = useCallback(async () => {
    if (!selectedDay) return;
    setBusy(true);
    try {
      await deleteWorkingHours(selectedDay);
      await reload();
      setSelectedDay(null);
    } finally {
      setBusy(false);
    }
  }, [selectedDay, reload]);

  return (
    <KioskScreen label="Schedule" testId="schedule-calendar">
      {/* Month navigation header */}
      <div className="schedule-calendar__header">
        <TouchButton
          variant="secondary"
          testId="calendar-prev-month"
          onClick={goPrevMonth}
        >
          ‹
        </TouchButton>
        <span className="schedule-calendar__month" data-testid="calendar-month-label">
          {MONTH_LABELS[view.month0]} {view.year}
        </span>
        <TouchButton
          variant="secondary"
          testId="calendar-next-month"
          onClick={goNextMonth}
        >
          ›
        </TouchButton>
      </div>

      {/* Month-to-date cost for the booth's AiPhoto-tagged resources. */}
      <div className="cost-summary" data-testid="cost-month-to-date">
        <span className="cost-summary__label">This month so far</span>
        <span className="cost-summary__amount" data-testid="cost-month-amount">
          {monthCost
            ? formatCost(monthCost.total, monthCost.currency)
            : monthCostError
              ? "—"
              : "…"}
        </span>
        {monthCostError && (
          <span className="cost-summary__note">{monthCostError}</span>
        )}
        <span className="cost-summary__hint" data-testid="cost-propagation-note">
          Costs are tagged “AiPhoto” and can take up to 24 hours to appear.
        </span>
      </div>

      {/* Weekday name row */}
      <div className="schedule-calendar__weekdays" aria-hidden="true">
        {WEEKDAY_LABELS.map((w) => (
          <span key={w} className="schedule-calendar__weekday">
            {w}
          </span>
        ))}
      </div>

      {/* Day grid */}
      <div className="schedule-calendar__grid" role="grid" data-testid="calendar-grid">
        {cells.map((cell, i) => {
          if (cell.day === null) {
            return <span key={`pad-${i}`} className="schedule-calendar__pad" />;
          }
          const day = cell.day;
          const hasHours = hoursByDay.has(day);
          const isToday = day === todayIso;
          const classes = [
            hasHours ? "schedule-calendar__day--defined" : "",
            isToday ? "schedule-calendar__day--today" : "",
          ]
            .filter(Boolean)
            .join(" ");
          return (
            <TouchButton
              key={day}
              role="gridcell"
              variant={selectedDay === day ? "primary" : "secondary"}
              className={classes || undefined}
              testId={`calendar-day-${day}`}
              data-has-hours={hasHours ? "true" : "false"}
              onClick={() => openEditor(day)}
            >
              <span className="schedule-calendar__daynum">{cell.dayNum}</span>
            </TouchButton>
          );
        })}
      </div>

      {selectedDay && (
        <div className="schedule-calendar__editor" data-testid="day-editor">
          <h2>{selectedDay}</h2>
          <label>
            Start
            <input
              type="time"
              value={startTime}
              data-testid="start-time-input"
              onChange={(e) => setStartTime(e.target.value)}
            />
          </label>
          <label>
            End
            <input
              type="time"
              value={endTime}
              data-testid="end-time-input"
              onChange={(e) => setEndTime(e.target.value)}
            />
          </label>

          {validationError && (
            <p role="alert" data-testid="schedule-validation-error">
              {validationError}
            </p>
          )}

          <div className="schedule-calendar__editor-actions">
            <TouchButton
              variant="secondary"
              testId="schedule-cancel-button"
              onClick={() => setSelectedDay(null)}
            >
              Cancel
            </TouchButton>
            {hoursByDay.has(selectedDay) && (
              <TouchButton
                variant="danger"
                testId="schedule-remove-button"
                disabled={busy}
                onClick={() => void onRemove()}
              >
                Remove
              </TouchButton>
            )}
            <PrimaryButton testId="schedule-save-button" disabled={busy} onClick={() => void onSave()}>
              Save
            </PrimaryButton>
          </div>

          {/* Per-day cost breakdown by service (component), filtered to the
              AiPhoto tag, plus the day's total. */}
          <div className="cost-breakdown" data-testid="cost-breakdown">
            <h3 className="cost-breakdown__title">Cost for {selectedDay}</h3>
            {dailyCostLoading && (
              <p className="cost-breakdown__status" data-testid="cost-day-loading">
                Loading cost…
              </p>
            )}
            {!dailyCostLoading && dailyCostError && (
              <p className="cost-breakdown__status" data-testid="cost-day-error">
                {dailyCostError}
              </p>
            )}
            {!dailyCostLoading && !dailyCostError && dailyCost && (
              dailyCost.components.length === 0 ? (
                <p className="cost-breakdown__status" data-testid="cost-day-empty">
                  No cost recorded for this day yet.
                </p>
              ) : (
                <ul className="cost-breakdown__list" data-testid="cost-day-list">
                  {dailyCost.components.map((c) => (
                    <li key={c.service} className="cost-breakdown__row">
                      <span className="cost-breakdown__service">{c.service}</span>
                      <span className="cost-breakdown__amount">
                        {formatCost(c.amount, dailyCost.currency)}
                      </span>
                    </li>
                  ))}
                  <li className="cost-breakdown__row cost-breakdown__row--total">
                    <span className="cost-breakdown__service">Total</span>
                    <span
                      className="cost-breakdown__amount"
                      data-testid="cost-day-total"
                    >
                      {formatCost(dailyCost.total, dailyCost.currency)}
                    </span>
                  </li>
                </ul>
              )
            )}
            <p className="cost-breakdown__hint">
              Costs can take up to 24 hours to appear.
            </p>
          </div>
        </div>
      )}
    </KioskScreen>
  );
}

export default ScheduleCalendar;
