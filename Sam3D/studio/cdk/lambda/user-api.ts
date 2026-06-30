/**
 * userApi — authenticated, non-admin routes:
 *   POST /upload-url   -> { jobId, uploadUrl, inputKey }  (presigned PUT for the request JSON)
 *   POST /generate     -> { outputKey, failureKey }       (InvokeEndpointAsync)
 *   GET  /result       -> { status, downloadUrl? | error? }
 *   GET  /endpoint      -> { status }                      (DescribeEndpoint)
 *
 * Every request is already authenticated by the API Gateway Cognito authorizer.
 * Large blobs (image/mask in, .ply out) never pass through here — only presigned
 * S3 URLs do. IAM on this function is scoped to the input/output prefixes and the
 * single endpoint ARN.
 */
import { randomUUID } from "crypto";
import { S3Client, PutObjectCommand, GetObjectCommand, HeadObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import {
  SageMakerRuntimeClient,
  InvokeEndpointAsyncCommand,
} from "@aws-sdk/client-sagemaker-runtime";
import { SageMakerClient, DescribeEndpointCommand } from "@aws-sdk/client-sagemaker";

const REGION = process.env.AWS_REGION!;
const BUCKET = process.env.IO_BUCKET!;
const INPUT_PREFIX = process.env.INPUT_PREFIX!;
const OUTPUT_PREFIX = process.env.OUTPUT_PREFIX!;
const FAILURE_PREFIX = process.env.FAILURE_PREFIX!;
const ENDPOINT = process.env.ENDPOINT_NAME!;
const TTL = parseInt(process.env.PRESIGN_TTL || "300", 10);

const s3 = new S3Client({ region: REGION });
const smr = new SageMakerRuntimeClient({ region: REGION });
const sm = new SageMakerClient({ region: REGION });

const json = (statusCode: number, body: unknown) => ({
  statusCode,
  headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
  body: JSON.stringify(body),
});

function safeKey(prefix: string, key: string | undefined): string | null {
  // Only allow keys under the expected prefix (prevents traversal / arbitrary reads).
  if (!key || key.includes("..") || !key.startsWith(prefix)) return null;
  return key;
}

export const handler = async (event: any) => {
  const method: string = event.httpMethod;
  const resource: string = event.resource || event.path || "";

  try {
    if (resource.endsWith("/upload-url") && method === "POST") {
      const jobId = randomUUID();
      const inputKey = `${INPUT_PREFIX}${jobId}.json`;
      const uploadUrl = await getSignedUrl(
        s3,
        new PutObjectCommand({ Bucket: BUCKET, Key: inputKey, ContentType: "application/json" }),
        { expiresIn: TTL }
      );
      return json(200, { jobId, uploadUrl, inputKey });
    }

    if (resource.endsWith("/generate") && method === "POST") {
      const body = JSON.parse(event.body || "{}");
      const inputKey = safeKey(INPUT_PREFIX, body.inputKey);
      if (!inputKey) return json(400, { error: "valid inputKey (under input prefix) is required" });

      const resp = await smr.send(
        new InvokeEndpointAsyncCommand({
          EndpointName: ENDPOINT,
          ContentType: "application/json",
          InputLocation: `s3://${BUCKET}/${inputKey}`,
          InvocationTimeoutSeconds: 3600,
        })
      );
      const toKey = (loc?: string) =>
        loc && loc.startsWith(`s3://${BUCKET}/`) ? loc.slice(`s3://${BUCKET}/`.length) : undefined;
      return json(200, {
        outputKey: toKey(resp.OutputLocation),
        failureKey: toKey(resp.FailureLocation),
      });
    }

    if (resource.endsWith("/result") && method === "GET") {
      const q = event.queryStringParameters || {};
      const outputKey = safeKey(OUTPUT_PREFIX, q.outputKey);
      const failureKey = q.failureKey ? safeKey(FAILURE_PREFIX, q.failureKey) : null;
      if (!outputKey) return json(400, { error: "valid outputKey (under output prefix) is required" });

      // done?
      try {
        await s3.send(new HeadObjectCommand({ Bucket: BUCKET, Key: outputKey }));
        const downloadUrl = await getSignedUrl(
          s3,
          new GetObjectCommand({ Bucket: BUCKET, Key: outputKey }),
          { expiresIn: TTL }
        );
        return json(200, { status: "done", downloadUrl });
      } catch (e: any) {
        if (e?.name !== "NotFound" && e?.$metadata?.httpStatusCode !== 404) throw e;
      }
      // failed?
      if (failureKey) {
        try {
          const obj = await s3.send(new GetObjectCommand({ Bucket: BUCKET, Key: failureKey }));
          const text = await obj.Body!.transformToString();
          return json(200, { status: "error", error: text.slice(0, 2000) });
        } catch (e: any) {
          if (e?.name !== "NoSuchKey" && e?.$metadata?.httpStatusCode !== 404) throw e;
        }
      }
      return json(200, { status: "pending" });
    }

    if (resource.endsWith("/endpoint") && method === "GET") {
      try {
        const d = await sm.send(new DescribeEndpointCommand({ EndpointName: ENDPOINT }));
        return json(200, { status: d.EndpointStatus, reason: d.FailureReason });
      } catch (e: any) {
        if (e?.name === "ValidationException" || e?.$metadata?.httpStatusCode === 400) {
          return json(200, { status: "Stopped" }); // endpoint doesn't exist
        }
        throw e;
      }
    }

    return json(404, { error: "not found" });
  } catch (err: any) {
    console.error(err);
    return json(500, { error: String(err?.message || err) });
  }
};
