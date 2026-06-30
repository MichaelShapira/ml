// Typed client for the API Gateway routes. Every call carries the Cognito ID
// token in the Authorization header (the Cognito authorizer validates it).
import { apiBase } from "./config";
import { getIdToken } from "./auth";

async function authed(path: string, init: RequestInit = {}): Promise<any> {
  const token = await getIdToken();
  const res = await fetch(apiBase() + path, {
    ...init,
    headers: { Authorization: token, "Content-Type": "application/json", ...(init.headers || {}) },
  });
  const text = await res.text();
  let data: any = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
  if (!res.ok) throw new Error(data?.error || `${res.status} ${res.statusText}`);
  return data;
}

export interface UploadTarget { jobId: string; uploadUrl: string; inputKey: string; }
export interface GenerateResult { outputKey: string; failureKey?: string; }
export interface ResultStatus { status: "pending" | "done" | "error"; downloadUrl?: string; error?: string; }
export interface EndpointStatus { status: string; reason?: string; message?: string; }

export const api = {
  uploadUrl: () => authed("upload-url", { method: "POST", body: "{}" }) as Promise<UploadTarget>,

  // Upload the request JSON straight to S3 with the presigned PUT (no auth header).
  async putInput(uploadUrl: string, payload: unknown): Promise<void> {
    const res = await fetch(uploadUrl, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`upload failed: ${res.status}`);
  },

  generate: (inputKey: string) =>
    authed("generate", { method: "POST", body: JSON.stringify({ inputKey }) }) as Promise<GenerateResult>,

  result: (outputKey: string, failureKey?: string) => {
    const q = new URLSearchParams({ outputKey });
    if (failureKey) q.set("failureKey", failureKey);
    return authed("result?" + q.toString()) as Promise<ResultStatus>;
  },

  endpointStatus: () => authed("endpoint") as Promise<EndpointStatus>,
  startEndpoint: () => authed("endpoint/start", { method: "POST", body: "{}" }) as Promise<EndpointStatus>,
  stopEndpoint: () => authed("endpoint/stop", { method: "POST", body: "{}" }) as Promise<EndpointStatus>,
};
