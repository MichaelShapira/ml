/**
 * CameraView — live webcam feed + still capture (Requirements 2, 3).
 *
 * Opens the webcam via `getUserMedia`, renders the live feed, and captures a
 * still to a base64 PNG data URL via a canvas (Requirement 3.4). Handles:
 *   - camera unavailable / permission denied → error message + Retry (Req 2.4);
 *   - capture temporarily unavailable → hide Take Photo + message (Req 2.3);
 *   - portrait layout (Req 2.5).
 *
 * On capture it calls `onCapture(dataUrl)`; the parent dispatches CAPTURE to
 * the state machine.
 */
import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import { KioskScreen, PrimaryButton, TouchButton } from "../theme";

export interface CameraViewProps {
  /** Called with the captured base64 image data URL. */
  onCapture: (photo: string) => void;
  /**
   * When true (admin mode), an "Upload Photo" control is shown so a photo can
   * be supplied from a file instead of the webcam. Available both alongside
   * Take Photo and on the camera-error screen, so an admin can proceed even if
   * the webcam is unavailable.
   */
  allowUpload?: boolean;
}

type CameraState = "starting" | "ready" | "error";

export function CameraView({ onCapture, allowUpload = false }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [state, setState] = useState<CameraState>("starting");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
  }, []);

  const startCamera = useCallback(async () => {
    setState("starting");
    setErrorMessage(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera API is not available in this browser.");
      }
      // Request the highest practical resolution. `ideal` is best-effort: the
      // browser picks the closest mode the camera actually supports and never
      // throws OverconstrainedError (unlike `min`/`exact`), so an HD webcam
      // yields 1080p/720p while a low-res one still works at its best mode.
      // Without these hints browsers default to ~640x480, which is what made
      // the captured photo look poor.
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {
          /* autoplay may require a gesture; the feed still renders */
        });
      }
      setState("ready");
    } catch {
      stopStream();
      setErrorMessage(
        "The camera cannot be accessed. Check the connection and permissions, then retry.",
      );
      setState("error");
    }
  }, [stopStream]);

  useEffect(() => {
    void startCamera();
    return () => stopStream();
  }, [startCamera, stopStream]);

  const takePhoto = useCallback(() => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) {
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    // Capture at the camera's full native resolution (canvas == videoWidth/
    // videoHeight) and keep rendering crisp.
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    // The live preview is mirrored (CSS `scaleX(-1)`) so it feels like a mirror.
    // Mirror the capture too so the saved photo matches what the visitor saw,
    // rather than flipping left/right at the moment of capture.
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, width, height);
    const dataUrl = canvas.toDataURL("image/png");
    onCapture(dataUrl);
  }, [onCapture]);

  // Admin-only: read a chosen image file as a base64 data URL and feed it into
  // the same Review → Effects → generation flow as a captured photo.
  const onFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      // Reset the input so selecting the same file again re-triggers change.
      event.target.value = "";
      if (!file || !file.type.startsWith("image/")) {
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          // Stop the live stream now that we have a photo from upload.
          stopStream();
          onCapture(reader.result);
        }
      };
      reader.readAsDataURL(file);
    },
    [onCapture, stopStream],
  );

  const openFilePicker = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  /** The hidden file input + an "Upload Photo" button, shown only in admin mode. */
  const uploadControl = allowUpload ? (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="camera-view__file-input"
        style={{ display: "none" }}
        data-testid="upload-photo-input"
        onChange={onFileChange}
      />
      <TouchButton
        variant="secondary"
        testId="upload-photo-button"
        onClick={openFilePicker}
      >
        Upload Photo
      </TouchButton>
    </>
  ) : null;

  return (
    <KioskScreen
      label="Camera"
      testId="camera-view"
      footer={
        state === "ready" ? (
          <div className="camera-view__actions">
            <PrimaryButton testId="take-photo-button" onClick={takePhoto}>
              Take Photo
            </PrimaryButton>
            {uploadControl}
          </div>
        ) : state === "error" ? (
          <div className="camera-view__actions">
            <PrimaryButton testId="retry-camera-button" onClick={() => void startCamera()}>
              Retry
            </PrimaryButton>
            {uploadControl}
          </div>
        ) : (
          uploadControl ?? undefined
        )
      }
    >
      {state === "error" ? (
        <p className="camera-view__error" role="alert" data-testid="camera-error">
          {errorMessage}
        </p>
      ) : (
        <>
          <video
            ref={videoRef}
            className="camera-view__video"
            playsInline
            muted
            autoPlay
            data-testid="camera-video"
          />
          {state === "starting" && (
            <p className="camera-view__status" data-testid="camera-unavailable">
              Camera is temporarily unavailable…
            </p>
          )}
        </>
      )}
    </KioskScreen>
  );
}

export default CameraView;
