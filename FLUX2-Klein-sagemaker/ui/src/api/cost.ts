/**
 * Cost_Service — read AWS spend for the booth's `AiPhoto`-tagged resources via
 * AWS Cost Explorer (`GetCostAndUsage`), filtered by the cost allocation tag.
 *
 * Two resolutions:
 *   - {@link getDailyCostByService}: a single day's `UnblendedCost`, broken down
 *     by AWS service (the "components"), filtered to `AiPhoto = true`.
 *   - {@link getMonthToDateCost}: the running total for the current month.
 *
 * Cost Explorer is global (endpoint in us-east-1) and the data lags by up to
 * ~24h, so the UI shows a clear "may take 24h to appear" note. The call runs
 * through {@link withAuthRetry} for silent token/STS refresh.
 *
 * NOTE: the `AiPhoto` cost allocation tag must be activated in Billing for the
 * tag filter to return data (the CDK best-effort-activates it on the management
 * account; otherwise an admin activates it once). Until usage is reported and
 * the tag is active, results can be empty/zero.
 */

import {
  GetCostAndUsageCommand,
  type Group,
} from "@aws-sdk/client-cost-explorer";
import { getCostExplorerClient, withAuthRetry } from "./awsClients";

/** The cost allocation tag key the whole stack is tagged with. */
export const COST_TAG_KEY = "AiPhoto";

/** A single cost line: one AWS service and its amount for the period. */
export interface CostComponent {
  /** AWS service name (e.g. "Amazon SageMaker"). */
  service: string;
  /** Cost amount in {@link currency} for the queried period. */
  amount: number;
}

/** A day's cost breakdown. */
export interface DailyCost {
  /** The `YYYY-MM-DD` day queried. */
  day: string;
  /** Per-service components, sorted by amount descending (zero ones dropped). */
  components: CostComponent[];
  /** Sum of all components for the day. */
  total: number;
  /** ISO currency code (e.g. "USD"). */
  currency: string;
}

/** A month-to-date total. */
export interface MonthCost {
  /** The `YYYY-MM` month queried. */
  month: string;
  /** Total cost for the month so far. */
  total: number;
  /** ISO currency code. */
  currency: string;
}

/** Add `days` days to a `YYYY-MM-DD` string, returning `YYYY-MM-DD` (UTC). */
function addDays(isoDay: string, days: number): string {
  const [y, m, d] = isoDay.split("-").map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d + days));
  return dt.toISOString().slice(0, 10);
}

/** First day of the month for a `YYYY-MM-DD` → `YYYY-MM-01`. */
function firstOfMonth(isoDay: string): string {
  const [y, m] = isoDay.split("-");
  return `${y}-${m}-01`;
}

/** The `AiPhoto = true` tag filter expression shared by both queries. */
function aiPhotoTagFilter() {
  return { Tags: { Key: COST_TAG_KEY, Values: ["true"] } };
}

/** Extract amount + currency from a Cost Explorer metric map. */
function readUnblended(metrics: Record<string, { Amount?: string; Unit?: string }> | undefined): {
  amount: number;
  currency: string;
} {
  const m = metrics?.UnblendedCost;
  return {
    amount: m?.Amount ? Number.parseFloat(m.Amount) : 0,
    currency: m?.Unit ?? "USD",
  };
}

/**
 * Daily cost for a single day, broken down by AWS service, filtered to the
 * booth's `AiPhoto` tag. Cost Explorer's DAILY granularity treats `End` as
 * exclusive, so we query `[day, day+1)`.
 */
export async function getDailyCostByService(day: string): Promise<DailyCost> {
  const ce = getCostExplorerClient();
  const response = await withAuthRetry(() =>
    ce.send(
      new GetCostAndUsageCommand({
        TimePeriod: { Start: day, End: addDays(day, 1) },
        Granularity: "DAILY",
        Metrics: ["UnblendedCost"],
        Filter: aiPhotoTagFilter(),
        GroupBy: [{ Type: "DIMENSION", Key: "SERVICE" }],
      }),
    ),
  );

  const result = response.ResultsByTime?.[0];
  const groups: Group[] = result?.Groups ?? [];
  let currency = "USD";
  const components: CostComponent[] = [];
  for (const g of groups) {
    const { amount, currency: cur } = readUnblended(g.Metrics);
    currency = cur;
    if (g.Keys?.[0] && amount > 0) {
      components.push({ service: g.Keys[0], amount });
    }
  }
  components.sort((a, b) => b.amount - a.amount);
  const total = components.reduce((sum, c) => sum + c.amount, 0);
  return { day, components, total, currency };
}

/**
 * Month-to-date total for the month containing `day` (defaults to today),
 * filtered to the booth's `AiPhoto` tag. `End` is exclusive, so we query from
 * the 1st through `day+1`.
 */
export async function getMonthToDateCost(day: string): Promise<MonthCost> {
  const ce = getCostExplorerClient();
  const start = firstOfMonth(day);
  const response = await withAuthRetry(() =>
    ce.send(
      new GetCostAndUsageCommand({
        TimePeriod: { Start: start, End: addDays(day, 1) },
        Granularity: "MONTHLY",
        Metrics: ["UnblendedCost"],
        Filter: aiPhotoTagFilter(),
      }),
    ),
  );

  // Sum across returned month buckets (normally one).
  let total = 0;
  let currency = "USD";
  for (const r of response.ResultsByTime ?? []) {
    const { amount, currency: cur } = readUnblended(r.Total);
    total += amount;
    currency = cur;
  }
  return { month: start.slice(0, 7), total, currency };
}

/** Format an amount + ISO currency for display (e.g. "$1.23"). */
export function formatCost(amount: number, currency: string): string {
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(amount);
  } catch {
    return `${amount.toFixed(2)} ${currency}`;
  }
}
