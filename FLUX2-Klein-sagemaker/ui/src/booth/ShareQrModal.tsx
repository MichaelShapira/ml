/**
 * ShareQrModal — the "Share with me" QR overlay.
 *
 * Replaces the old email form. Given the in-flight share state, it shows:
 *   - while uploading + presigning: a spinner ("Preparing your download…");
 *   - on success: a large, crisp SVG QR code of the 15-minute presigned URL,
 *     layout-aware guidance, and "Valid for 15 minutes";
 *   - on failure: an error message.
 *
 * Why SVG + these settings (the old raster QR didn't scan):
 *   - Presigned S3 URLs are long (~700+ chars), so the QR is dense. We render a
 *     crisp **SVG** (no scaling blur), a **4-module quiet zone** (`marginSize`),
 *     and error-correction level **"L"** — for long data, lower EC means fewer
 *     modules, i.e. larger modules at the same size, which scans far better than
 *     a small, high-EC raster image. High contrast (black on white) throughout.
 *
 * Layout-aware:
 *   - "monitor" (kiosk): scan-only. The visitor scans with their phone; there
 *     is no "click to download" (the kiosk can't receive the file).
 *   - "mobile": the visitor can EITHER scan the code with another phone OR tap
 *     the Download button to save it on this device. "Click to download" only
 *     appears here.
 */
import { QRCodeSVG } from "qrcode.react";
import { TouchButton } from "../theme";
import type { LayoutMode } from "./useLayoutMode";

/** The share state the modal renders. `idle` means the modal is not shown. */
export type ShareState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; url: string; expiresInSeconds: number }
  | { status: "error"; message: string };

export interface ShareQrModalProps {
  /** Current share state. The modal renders for loading/ready/error. */
  state: ShareState;
  /** Layout: drives whether "tap to download" is offered (mobile only). */
  layout: LayoutMode;
  /** Close the modal (back to idle). */
  onClose: () => void;
}

/** Format a TTL in seconds as a human "N minutes" string. */
function formatMinutes(seconds: number): string {
  const minutes = Math.max(1, Math.round(seconds / 60));
  return `${minutes} minute${minutes === 1 ? "" : "s"}`;
}

export function ShareQrModal({ state, layout, onClose }: ShareQrModalProps) {
  if (state.status === "idle") {
    return null;
  }

  const isMobile = layout === "mobile";
  // A generous on-screen size so the dense (long presigned URL) QR has large
  // modules. Level "L" keeps the module count as low as possible for long data.
  const qrSize = isMobile ? 300 : 440;

  return (
    <div
      className="share-modal__backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Share your photo"
      data-testid="share-modal"
      onClick={onClose}
    >
      {/* Stop propagation so taps inside the card don't close the modal. */}
      <div className="share-modal__card" onClick={(e) => e.stopPropagation()}>
        {state.status === "loading" && (
          <div className="share-modal__loading" role="status" aria-live="polite">
            <div className="loading-screen__spinner" aria-hidden="true" />
            <p data-testid="share-loading">Preparing your download…</p>
          </div>
        )}

        {state.status === "ready" && (
          <div className="share-modal__ready">
            <h2 className="share-modal__title">Your photo is ready</h2>

            <div className="share-modal__qr" data-testid="share-qr">
              <QRCodeSVG
                value={state.url}
                size={qrSize}
                level="L"
                marginSize={4}
                bgColor="#ffffff"
                fgColor="#000000"
                title="QR code linking to your photo download"
              />
            </div>

            {/* Layout-aware guidance. Make clear scanning AND/OR downloading. */}
            <p className="share-modal__instruction" data-testid="share-instruction">
              {isMobile
                ? "Scan the code with another phone, or tap Download to save it here."
                : "Scan the code with your phone to download your photo."}
            </p>

            {/* "Click to download" only on smartphone mode. */}
            {isMobile && (
              <a
                href={state.url}
                target="_blank"
                rel="noopener noreferrer"
                className="share-modal__download"
                data-testid="share-download"
              >
                Click to download
              </a>
            )}

            <p className="share-modal__ttl" data-testid="share-ttl">
              Valid for {formatMinutes(state.expiresInSeconds)}
            </p>
          </div>
        )}

        {state.status === "error" && (
          <p className="share-modal__error" role="alert" data-testid="share-error">
            {state.message}
          </p>
        )}

        <TouchButton variant="secondary" testId="share-close-button" onClick={onClose}>
          Close
        </TouchButton>
      </div>
    </div>
  );
}

export default ShareQrModal;
