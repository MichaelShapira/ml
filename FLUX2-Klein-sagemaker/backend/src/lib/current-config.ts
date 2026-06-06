/**
 * Current-endpoint-config pointer — pure key/shape logic (no AWS SDK, no I/O).
 *
 * The booth no longer hardcodes a single SageMaker endpoint config. Instead the
 * admin picks the **current** config from the account's configs, and the booth
 * invokes the endpoint created from that config. The choice is persisted as a
 * single item in the Schedule_Store table so the browser, the invoke proxy, and
 * the scheduler all agree on which endpoint is "current":
 *
 *   pk = "CONFIG#CURRENT", sk = "CONFIG#CURRENT", configName = "<chosen config>"
 *
 * A single fixed-key item guarantees exactly one config can be current at a
 * time — writing a new value overwrites the previous one (so making B current
 * automatically un-currents A).
 *
 * The endpoint created from a config is named identically to the config, so the
 * "current config name" doubles as the "current endpoint name" the booth drives.
 */

/** Fixed partition/sort key value for the single current-config pointer. */
export const CURRENT_CONFIG_PK = "CONFIG#CURRENT";
export const CURRENT_CONFIG_SK = "CONFIG#CURRENT";

/**
 * Stable "endpoint name" used as the Working_Hours partition key, independent
 * of which config is current. The schedule is booth-wide (it applies to
 * whichever config is current), so it must NOT be keyed by the changing config
 * name — otherwise switching the current config would orphan the schedule.
 */
export const BOOTH_SCHEDULE_NAME = "booth";

/** The persisted current-config pointer item shape. */
export interface CurrentConfigItem {
  /** Always {@link CURRENT_CONFIG_PK}. */
  pk: string;
  /** Always {@link CURRENT_CONFIG_SK}. */
  sk: string;
  /** The chosen SageMaker endpoint configuration name (and endpoint name). */
  configName: string;
  /** ISO timestamp of the last write. */
  updatedAt: string;
}

/** Build the DynamoDB key object for the current-config pointer. */
export function currentConfigKey(): { pk: string; sk: string } {
  return { pk: CURRENT_CONFIG_PK, sk: CURRENT_CONFIG_SK };
}

/** Build a full current-config item for a chosen config name. */
export function makeCurrentConfigItem(configName: string): CurrentConfigItem {
  return {
    pk: CURRENT_CONFIG_PK,
    sk: CURRENT_CONFIG_SK,
    configName,
    updatedAt: new Date().toISOString(),
  };
}
