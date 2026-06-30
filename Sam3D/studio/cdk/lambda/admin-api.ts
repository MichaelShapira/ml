/**
 * adminApi — admin-only endpoint lifecycle control:
 *   POST /endpoint/start -> recreate the endpoint from its existing config
 *   POST /endpoint/stop  -> delete the endpoint (stops all GPU billing)
 *
 * Authorization is enforced HERE (defense in depth): the API Gateway Cognito
 * authorizer guarantees the caller is authenticated; this function additionally
 * requires the JWT claim custom:role === "admin" and returns 403 otherwise. So
 * the route is never admin-open by gateway config alone.
 *
 * "start" recreates the SageMaker endpoint from the pre-existing endpoint-config
 * (the model + BYOC image + weights live in that config, created by the
 * notebook). "stop" deletes the endpoint resource — the config is untouched, so
 * a later "start" brings it back. IAM is scoped to the single endpoint/config.
 */
import {
  SageMakerClient,
  CreateEndpointCommand,
  DeleteEndpointCommand,
  DescribeEndpointCommand,
} from "@aws-sdk/client-sagemaker";

const REGION = process.env.AWS_REGION!;
const ENDPOINT = process.env.ENDPOINT_NAME!;
const ENDPOINT_CONFIG = process.env.ENDPOINT_CONFIG_NAME!;

const sm = new SageMakerClient({ region: REGION });

const json = (statusCode: number, body: unknown) => ({
  statusCode,
  headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
  body: JSON.stringify(body),
});

function isAdmin(event: any): boolean {
  // REST API + Cognito authorizer puts claims here:
  const claims = event?.requestContext?.authorizer?.claims || {};
  return claims["custom:role"] === "admin";
}

async function currentStatus(): Promise<string> {
  try {
    const d = await sm.send(new DescribeEndpointCommand({ EndpointName: ENDPOINT }));
    return d.EndpointStatus || "Unknown";
  } catch (e: any) {
    if (e?.name === "ValidationException" || e?.$metadata?.httpStatusCode === 400) return "Stopped";
    throw e;
  }
}

export const handler = async (event: any) => {
  if (!isAdmin(event)) {
    return json(403, { error: "admin role required" });
  }
  const resource: string = event.resource || event.path || "";

  try {
    if (resource.endsWith("/endpoint/start")) {
      const status = await currentStatus();
      if (status !== "Stopped") {
        return json(200, { status, message: `endpoint already ${status}` });
      }
      await sm.send(
        new CreateEndpointCommand({
          EndpointName: ENDPOINT,
          EndpointConfigName: ENDPOINT_CONFIG,
        })
      );
      return json(200, { status: "Creating", message: "endpoint is starting (cold start ~minutes)" });
    }

    if (resource.endsWith("/endpoint/stop")) {
      const status = await currentStatus();
      if (status === "Stopped") {
        return json(200, { status: "Stopped", message: "endpoint already stopped" });
      }
      await sm.send(new DeleteEndpointCommand({ EndpointName: ENDPOINT }));
      return json(200, { status: "Stopping", message: "endpoint is being deleted (GPU billing stops)" });
    }

    return json(404, { error: "not found" });
  } catch (err: any) {
    console.error(err);
    return json(500, { error: String(err?.message || err) });
  }
};
