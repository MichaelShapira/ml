/**
 * Endpoint regions: where the booth's SageMaker endpoints may run.
 *
 * The booth itself (Cognito, DynamoDB, CloudFront, Lambdas) lives in the
 * stack's region, the "home" region. An admin may also add endpoint configs
 * from the extra regions the stack was deployed with (deploy.sh
 * `--endpoint-regions`), for example to use GPU capacity that the home region
 * does not have. Each endpoint region has its own async I/O bucket: SageMaker
 * reads model data in the endpoint's region, and the booth keeps requests and
 * results next to the endpoint.
 *
 * The list comes from the runtime config (`endpointRegions`, written by CDK).
 * Without it, the home region with the stack's I/O bucket is the only one.
 */
import { getConfig } from "../config";

/** A region the booth may run an endpoint in. */
export interface EndpointRegion {
  /** AWS region name, e.g. `us-east-2`. */
  region: string;
  /** Async I/O bucket in that region (inputs/outputs/failures prefixes). */
  ioBucket: string;
  /** True for the stack's own region. */
  home: boolean;
}

/** An endpoint config and the region it lives in (names are unique per region only). */
export interface ConfigRef {
  name: string;
  region: string;
}

/** Thrown when a config or request points at a region the booth is not set up for. */
export class UnknownRegionError extends Error {
  constructor(public readonly region: string) {
    super(`Region "${region}" is not one of the booth's endpoint regions.`);
    this.name = "UnknownRegionError";
  }
}

/** The stack's region (Cognito, DynamoDB and the original endpoint). */
export function homeRegion(): string {
  return getConfig().region;
}

/** Every endpoint region, home first. */
export function getEndpointRegions(): EndpointRegion[] {
  const config = getConfig();
  const regions: EndpointRegion[] = [
    { region: config.region, ioBucket: config.ioBucket, home: true },
  ];
  for (const extra of config.endpointRegions ?? []) {
    if (
      extra &&
      typeof extra.region === "string" &&
      typeof extra.ioBucket === "string" &&
      extra.region.length > 0 &&
      extra.ioBucket.length > 0 &&
      !regions.some((r) => r.region === extra.region)
    ) {
      regions.push({ region: extra.region, ioBucket: extra.ioBucket, home: false });
    }
  }
  return regions;
}

/** `region`, or the home region when it is missing (rows written before regions existed). */
export function regionOrHome(region: string | null | undefined): string {
  return typeof region === "string" && region.length > 0 ? region : homeRegion();
}

/** The async I/O bucket for an endpoint region. */
export function ioBucketForRegion(region: string | null | undefined): string {
  const name = regionOrHome(region);
  const match = getEndpointRegions().find((r) => r.region === name);
  if (!match) {
    throw new UnknownRegionError(name);
  }
  return match.ioBucket;
}

/** True when `region` is the stack's region (or missing). */
export function isHomeRegion(region: string | null | undefined): boolean {
  return regionOrHome(region) === homeRegion();
}

/**
 * Stable id of a config across regions, used for React keys and test ids: the
 * bare name in the home region (unchanged from before regions existed), and
 * `name@region` elsewhere.
 */
export function configId(ref: ConfigRef): string {
  return isHomeRegion(ref.region) ? ref.name : `${ref.name}@${ref.region}`;
}
