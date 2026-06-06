/**
 * Email_Service — send the visitor's transformed photo by email (SES v2).
 *
 * The booth has no backend, so the browser sends the email directly using
 * Amazon SES v2 `SendEmail` with a **raw MIME** message under the visitor's
 * Identity-Pool credentials. Raw MIME lets us attach the actual PNG (rather
 * than a link), so the recipient gets the image itself.
 *
 * The call runs through {@link withAuthRetry} for silent token/STS refresh.
 *
 * SES sandbox note: until the account has SES production access, the recipient
 * address must be a verified identity. The sender (From) is the verified
 * identity provisioned by the CDK SES construct (`config.senderEmail`).
 */

import { SendEmailCommand } from "@aws-sdk/client-sesv2";
import { getConfig } from "../config";
import { getSesClient } from "./awsClients";

/** A minimal, permissive email-shape check (full RFC validation isn't useful). */
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/** Returns true iff `email` looks like a usable address. */
export function isValidEmail(email: string): boolean {
  return EMAIL_RE.test(email.trim());
}

/** Thrown when the supplied recipient address is not a valid email. */
export class InvalidEmailError extends Error {
  constructor(message = "Please enter a valid email address.") {
    super(message);
    this.name = "InvalidEmailError";
  }
}

/** Strip a `data:` URL prefix, returning just the base64 payload. */
function base64FromDataUrl(dataUrl: string): string {
  const comma = dataUrl.indexOf(",");
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

/**
 * Read an image (object URL or data URL) into a base64 PNG string suitable for
 * a MIME attachment.
 */
async function imageToBase64(imageSrc: string): Promise<string> {
  if (imageSrc.startsWith("data:")) {
    return base64FromDataUrl(imageSrc);
  }
  // blob: / http(s): object URL → fetch the bytes and base64-encode them.
  const response = await fetch(imageSrc);
  const blob = await response.blob();
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/** Split a base64 string into fixed-width lines (RFC 2045 recommends ≤ 76). */
function chunkBase64(b64: string, width = 76): string {
  const lines: string[] = [];
  for (let i = 0; i < b64.length; i += width) {
    lines.push(b64.slice(i, i + width));
  }
  return lines.join("\r\n");
}

/** Build a multipart/mixed raw MIME message with the PNG attached. */
function buildRawMessage(params: {
  from: string;
  to: string;
  subject: string;
  bodyText: string;
  pngBase64: string;
  filename: string;
}): string {
  const boundary = `=_booth_${Date.now().toString(36)}_${Math.random()
    .toString(36)
    .slice(2)}`;
  const attachment = chunkBase64(params.pngBase64);

  return [
    `From: ${params.from}`,
    `To: ${params.to}`,
    `Subject: ${params.subject}`,
    "MIME-Version: 1.0",
    `Content-Type: multipart/mixed; boundary="${boundary}"`,
    "",
    `--${boundary}`,
    'Content-Type: text/plain; charset="UTF-8"',
    "Content-Transfer-Encoding: 7bit",
    "",
    params.bodyText,
    "",
    `--${boundary}`,
    'Content-Type: image/png; name="' + params.filename + '"',
    "Content-Transfer-Encoding: base64",
    `Content-Disposition: attachment; filename="${params.filename}"`,
    "",
    attachment,
    "",
    `--${boundary}--`,
    "",
  ].join("\r\n");
}

/** Input to {@link sendPhotoEmail}. */
export interface SendPhotoEmailInput {
  /** Recipient email address. */
  to: string;
  /** The transformed image source — an object URL or a data URL. */
  imageSrc: string;
  /** Optional subject override. */
  subject?: string;
}

/**
 * Email the visitor's AI-generated photo as a PNG attachment via SES v2
 * (Requirement: result-screen email delivery).
 *
 * @throws {InvalidEmailError} when `to` is not a valid email address.
 * @throws Error on an SES failure (e.g. sandbox: unverified recipient).
 */
export async function sendPhotoEmail(input: SendPhotoEmailInput): Promise<void> {
  const to = input.to.trim();
  if (!isValidEmail(to)) {
    throw new InvalidEmailError();
  }

  const config = getConfig();
  const pngBase64 = await imageToBase64(input.imageSrc);
  const subject = input.subject ?? "Your AI Photo Booth photo";
  const bodyText =
    "Thanks for visiting the AI Photo Booth! Your AI-generated photo is attached. " +
    "This image was created with AI.";

  const raw = buildRawMessage({
    from: config.senderEmail,
    to,
    subject,
    bodyText,
    pngBase64,
    filename: "ai-photo-booth.png",
  });

  // Encode the raw MIME string to bytes for the SES Raw.Data field.
  const data = new TextEncoder().encode(raw);

  const ses = getSesClient();
  try {
    await ses.send(
      new SendEmailCommand({
        FromEmailAddress: config.senderEmail,
        Destination: { ToAddresses: [to] },
        Content: { Raw: { Data: data } },
      }),
    );
  } catch (err) {
    // A failed email must NOT sign the visitor out. SES rejections surface as
    // 403 AccessDenied (e.g. sandbox: unverified recipient) or MessageRejected;
    // those are delivery problems, not an expired session, so we translate them
    // into a friendly EmailDeliveryError instead of routing through the
    // auth-retry/sign-in path that a generic 403 would trigger.
    throw toEmailError(err);
  }
}

/** Thrown when SES rejects the message (delivery problem, not an auth problem). */
export class EmailDeliveryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EmailDeliveryError";
  }
}

/** Map an SES SDK error into a friendly, non-auth {@link EmailDeliveryError}. */
function toEmailError(err: unknown): Error {
  const e = (err ?? {}) as {
    name?: string;
    message?: string;
    $metadata?: { httpStatusCode?: number };
  };
  const name = e.name ?? "";
  const msg = e.message ?? "";
  // Sandbox / unverified recipient is the most common cause for the booth.
  if (
    /MessageRejected/i.test(name) ||
    /not verified/i.test(msg) ||
    (/AccessDenied/i.test(name) && /ses:/i.test(msg))
  ) {
    return new EmailDeliveryError(
      "Couldn't send to that address. While the booth is in email test mode, " +
        "only pre-approved recipients can receive photos.",
    );
  }
  return new EmailDeliveryError(
    "Couldn't send the email right now. Please try again.",
  );
}
