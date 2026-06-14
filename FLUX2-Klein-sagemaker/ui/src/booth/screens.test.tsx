/**
 * Unit tests for the capture-flow screens (Requirements 1, 2, 4, 5, 6, 10).
 *
 * These cover the presentational screens in isolation (the full flow + AWS
 * wiring lives in BoothFlow). Verifies:
 *   - Start shows a single primary control (Req 1.1);
 *   - Camera shows Take Photo when ready and an error + retry on failure
 *     (Req 2.3, 2.4), and capture encodes a base64 image (Req 3.4);
 *   - Review shows the photo + reset/continue (Req 4.3);
 *   - StudioView shows all 12 options + the original photo + membership labels
 *     (Req 5.2/5.3/6.2/6.3), locks options while loading, and shows the result
 *     + AI-generated notice + New Session (Req 10.2, 10.3).
 */
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

// StudioView imports ../api/share (which pulls in the AWS client chain).
// Stub it so these presentational tests don't instantiate AWS clients/config.
vi.mock("../api/share", () => ({
  shareImage: vi.fn().mockResolvedValue({
    url: "https://example.com/presigned",
    expiresInSeconds: 900,
  }),
  SHARE_TTL_SECONDS: 900,
}));

import { StartScreen } from "./StartScreen";
import { ReviewScreen } from "./ReviewScreen";
import { StudioView } from "./StudioView";
import { CameraView } from "./CameraView";
import { EFFECTS } from "./effects";

describe("StartScreen (Req 1.1)", () => {
  it("renders a single Start control", () => {
    const onStart = vi.fn();
    render(<StartScreen onStart={onStart} />);
    const button = screen.getByTestId("start-button");
    fireEvent.click(button);
    expect(onStart).toHaveBeenCalledOnce();
  });
});

describe("ReviewScreen (Req 4.3)", () => {
  it("shows the captured photo with Reset and Continue", () => {
    const onReset = vi.fn();
    const onContinue = vi.fn();
    render(
      <ReviewScreen photo="data:image/png;base64,AAA" onReset={onReset} onContinue={onContinue} />,
    );
    expect((screen.getByTestId("review-photo") as HTMLImageElement).src).toContain(
      "data:image/png;base64,AAA",
    );
    fireEvent.click(screen.getByTestId("reset-button"));
    fireEvent.click(screen.getByTestId("continue-button"));
    expect(onReset).toHaveBeenCalledOnce();
    expect(onContinue).toHaveBeenCalledOnce();
  });
});

describe("StudioView effects (Req 5.2/5.3/6.2/6.3, 5.6, 4.3)", () => {
  const baseProps = {
    layout: "mobile" as const,
    photo: "data:image/png;base64,AAA",
    onSelect: vi.fn(),
    onMerge: vi.fn(),
    onNewSession: vi.fn(),
  };

  it("renders all 12 options, the original photo, and the named effect labels", () => {
    render(<StudioView {...baseProps} phase="idle" />);

    // The original photo is shown alongside the options (mobile carousel image).
    expect(screen.getByTestId("carousel-image")).toBeInTheDocument();

    // All catalog options are rendered.
    for (const effect of EFFECTS) {
      expect(screen.getByTestId(`effect-${effect.id}`)).toBeInTheDocument();
    }
    expect(screen.getAllByRole("listitem")).toHaveLength(EFFECTS.length);

    // Spot-check required membership labels.
    expect(screen.getByText("Spaceship interior")).toBeInTheDocument();
    expect(screen.getByText("Roman colosseum")).toBeInTheDocument();
    expect(screen.getByText("Viking warrior")).toBeInTheDocument();
    expect(screen.getByText("Astronaut")).toBeInTheDocument();
  });

  it("records a selection and disables all options while loading", () => {
    const onSelect = vi.fn();
    const { rerender } = render(
      <StudioView {...baseProps} phase="idle" onSelect={onSelect} />,
    );
    fireEvent.click(screen.getByTestId("effect-bg_spaceship"));
    expect(onSelect).toHaveBeenCalledWith("bg_spaceship", "background");

    rerender(
      <StudioView
        {...baseProps}
        phase="loading"
        loadingMessage="Brewing the paints…"
        onSelect={onSelect}
      />,
    );
    expect(screen.getByTestId("effect-person_viking")).toBeDisabled();
    // Loading message is surfaced.
    expect(screen.getByTestId("loading-message")).toBeInTheDocument();
  });
});

describe("StudioView result (Req 10.2, 10.3)", () => {
  it("shows the generated image, the AI-generated notice, and New Session", () => {
    const onNewSession = vi.fn();
    render(
      <StudioView
        layout="monitor"
        photo="data:image/png;base64,AAA"
        phase="result"
        results={{ person: "blob:abc" }}
        onSelect={vi.fn()}
        onMerge={vi.fn()}
        onNewSession={onNewSession}
      />,
    );
    expect(screen.getByTestId("result-image-person")).toBeInTheDocument();
    expect(screen.getByTestId("ai-generated-notice")).toHaveTextContent(/ai-generated/i);
    fireEvent.click(screen.getByTestId("new-session-button"));
    expect(onNewSession).toHaveBeenCalledOnce();
  });

  it("mobile result offers an original/generated carousel toggle", () => {
    render(
      <StudioView
        layout="mobile"
        photo="data:image/png;base64,AAA"
        phase="result"
        results={{ person: "blob:abc" }}
        onSelect={vi.fn()}
        onMerge={vi.fn()}
        onNewSession={vi.fn()}
      />,
    );
    // Both carousel tabs are present so the visitor can compare.
    expect(screen.getByTestId("carousel-tab-original")).toBeInTheDocument();
    expect(screen.getByTestId("carousel-tab-person")).toBeInTheDocument();
    // Defaults to showing the newest generated image.
    expect(
      (screen.getByTestId("carousel-image") as HTMLImageElement).src,
    ).toContain("blob:abc");
    // Toggling to Original swaps the image.
    fireEvent.click(screen.getByTestId("carousel-tab-original"));
    expect(
      (screen.getByTestId("carousel-image") as HTMLImageElement).src,
    ).toContain("data:image/png;base64,AAA");
  });

  it("shows a Merge button only once both background and character images exist", () => {
    const onMerge = vi.fn();
    const props = {
      layout: "monitor" as const,
      photo: "data:image/png;base64,AAA",
      phase: "result" as const,
      onSelect: vi.fn(),
      onNewSession: vi.fn(),
    };
    const { rerender } = render(
      <StudioView {...props} results={{ background: "blob:bg" }} onMerge={onMerge} />,
    );
    // Only one source image so far: no merge offered.
    expect(screen.queryByTestId("merge-button")).not.toBeInTheDocument();

    rerender(
      <StudioView
        {...props}
        results={{ background: "blob:bg", person: "blob:person" }}
        onMerge={onMerge}
      />,
    );
    const mergeBtn = screen.getByTestId("merge-button");
    fireEvent.click(mergeBtn);
    expect(onMerge).toHaveBeenCalledOnce();
  });

  it("offers a 'Share with me' button per generated image and opens the QR modal on click", async () => {
    render(
      <StudioView
        layout="monitor"
        photo="data:image/png;base64,AAA"
        phase="result"
        results={{ person: "blob:abc" }}
        onSelect={vi.fn()}
        onMerge={vi.fn()}
        onNewSession={vi.fn()}
      />,
    );
    // A share button sits next to the generated image.
    const shareBtn = screen.getByTestId("share-button-person");
    fireEvent.click(shareBtn);
    // The QR modal opens and, once the (mocked) presign resolves, shows the
    // scan instruction and the 15-minute validity. On the monitor (kiosk) there
    // is no "click to download" — that is smartphone-only.
    expect(await screen.findByTestId("share-modal")).toBeInTheDocument();
    expect(await screen.findByTestId("share-ttl")).toHaveTextContent(/15 minutes/i);
    expect(screen.getByTestId("share-instruction")).toHaveTextContent(/scan/i);
    expect(screen.queryByTestId("share-download")).not.toBeInTheDocument();
  });

  it("offers a tap-to-download link in the share modal on mobile (smartphone) only", async () => {
    render(
      <StudioView
        layout="mobile"
        photo="data:image/png;base64,AAA"
        phase="result"
        results={{ person: "blob:abc" }}
        onSelect={vi.fn()}
        onMerge={vi.fn()}
        onNewSession={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("share-button-person"));
    expect(await screen.findByTestId("share-modal")).toBeInTheDocument();
    // Smartphone: both scan AND tap-to-download are offered.
    expect(await screen.findByTestId("share-download")).toHaveTextContent(/download/i);
    expect(screen.getByTestId("share-instruction")).toHaveTextContent(/scan|download/i);
  });
});

describe("CameraView (Req 2.3, 2.4, 3.4)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a camera error message with a retry control when access fails", async () => {
    // getUserMedia rejects → error state.
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(new Error("denied")) },
    });
    render(<CameraView onCapture={vi.fn()} />);
    expect(await screen.findByTestId("camera-error")).toBeInTheDocument();
    expect(screen.getByTestId("retry-camera-button")).toBeInTheDocument();
  });

  it("captures a base64 image when ready", async () => {
    const onCapture = vi.fn();
    // Provide a fake stream so the component reaches the ready state.
    const fakeStream = { getTracks: () => [] } as unknown as MediaStream;
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue(fakeStream) },
    });
    // Stub canvas → data URL and video dimensions.
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: vi.fn(),
      translate: vi.fn(),
      scale: vi.fn(),
    } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, "toDataURL").mockReturnValue(
      "data:image/png;base64,ZZZ",
    );
    Object.defineProperty(HTMLVideoElement.prototype, "videoWidth", {
      configurable: true,
      get: () => 640,
    });
    Object.defineProperty(HTMLVideoElement.prototype, "videoHeight", {
      configurable: true,
      get: () => 480,
    });
    Object.defineProperty(HTMLVideoElement.prototype, "play", {
      configurable: true,
      value: vi.fn().mockResolvedValue(undefined),
    });

    render(<CameraView onCapture={onCapture} />);
    const takeBtn = await screen.findByTestId("take-photo-button");
    fireEvent.click(takeBtn);
    expect(onCapture).toHaveBeenCalledWith("data:image/png;base64,ZZZ");
  });

  it("does not show the Upload Photo control unless allowUpload is set", async () => {
    const fakeStream = { getTracks: () => [] } as unknown as MediaStream;
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue(fakeStream) },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "play", {
      configurable: true,
      value: vi.fn().mockResolvedValue(undefined),
    });
    render(<CameraView onCapture={vi.fn()} />);
    await screen.findByTestId("take-photo-button");
    expect(screen.queryByTestId("upload-photo-button")).not.toBeInTheDocument();
  });

  it("admin upload (allowUpload) reads a file and calls onCapture with its data URL", async () => {
    const onCapture = vi.fn();
    // getUserMedia fails so we're on the error screen — admin can still upload.
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(new Error("no camera")) },
    });
    render(<CameraView onCapture={onCapture} allowUpload />);
    await screen.findByTestId("camera-error");

    const input = screen.getByTestId("upload-photo-input") as HTMLInputElement;
    expect(screen.getByTestId("upload-photo-button")).toBeInTheDocument();

    const file = new File(["bytes"], "me.png", { type: "image/png" });
    // jsdom's FileReader.readAsDataURL works; await the async onload.
    fireEvent.change(input, { target: { files: [file] } });
    await waitFor(() =>
      expect(onCapture).toHaveBeenCalledWith(
        expect.stringMatching(/^data:image\/png;base64,/),
      ),
    );
  });
});
