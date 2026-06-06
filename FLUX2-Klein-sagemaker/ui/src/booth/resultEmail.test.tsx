/**
 * Tests for the studio "email my photo" feature (SES v2 path).
 *
 * Mocks `../api/email` so the form logic is exercised without AWS:
 *   - a valid address calls sendPhotoEmail and shows a success message;
 *   - an invalid address shows a validation error and does not send;
 *   - a send failure surfaces an error message.
 *
 * The email form lives inside StudioView, shown once a result exists
 * (phase="result"). We render the studio in monitor layout for these tests.
 */
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

const sendPhotoEmail = vi.fn();

vi.mock("../api/email", () => ({
  sendPhotoEmail: (...a: unknown[]) => sendPhotoEmail(...a),
  // Use the real-ish validity check so the form's gate behaves correctly.
  isValidEmail: (e: string) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e.trim()),
  InvalidEmailError: class extends Error {},
  EmailDeliveryError: class extends Error {},
}));

import { StudioView } from "./StudioView";

function renderResult() {
  return render(
    <StudioView
      layout="monitor"
      photo="data:image/png;base64,AAA"
      phase="result"
      generatedUrl="blob:abc"
      onSelect={vi.fn()}
      onNewSession={vi.fn()}
    />,
  );
}

beforeEach(() => sendPhotoEmail.mockReset());

describe("Studio email feature", () => {
  it("sends the photo to a valid address and shows a success message", async () => {
    sendPhotoEmail.mockResolvedValue(undefined);
    renderResult();

    fireEvent.change(screen.getByTestId("email-input"), {
      target: { value: "visitor@example.com" },
    });
    fireEvent.click(screen.getByTestId("send-email-button"));

    await waitFor(() => expect(screen.getByTestId("email-sent")).toBeInTheDocument());
    expect(sendPhotoEmail).toHaveBeenCalledWith({
      to: "visitor@example.com",
      imageSrc: "blob:abc",
    });
  });

  it("rejects an invalid address without sending", async () => {
    renderResult();

    fireEvent.change(screen.getByTestId("email-input"), {
      target: { value: "not-an-email" },
    });
    fireEvent.click(screen.getByTestId("send-email-button"));

    expect(await screen.findByTestId("email-error")).toBeInTheDocument();
    expect(sendPhotoEmail).not.toHaveBeenCalled();
  });

  it("surfaces a send failure as an error message", async () => {
    sendPhotoEmail.mockRejectedValueOnce(new Error("SES failed"));
    renderResult();

    fireEvent.change(screen.getByTestId("email-input"), {
      target: { value: "visitor@example.com" },
    });
    fireEvent.click(screen.getByTestId("send-email-button"));

    expect(await screen.findByTestId("email-error")).toBeInTheDocument();
  });

  it("shows an on-screen keyboard on the monitor layout and builds the address from key taps", async () => {
    sendPhotoEmail.mockResolvedValue(undefined);
    renderResult();

    const input = screen.getByTestId("email-input") as HTMLInputElement;
    // No native keyboard on the kiosk monitor: the field is read-only and our
    // on-screen keyboard appears on focus.
    expect(input).toHaveAttribute("readonly");
    fireEvent.focus(input);
    expect(screen.getByTestId("email-keyboard")).toBeInTheDocument();

    // Tap keys to build "a@b.com" using a domain shortcut.
    fireEvent.click(screen.getByTestId("key-a"));
    fireEvent.click(screen.getByTestId("key-symbol-@"));
    fireEvent.click(screen.getByTestId("key-b"));
    fireEvent.click(screen.getByTestId("key-shortcut-.com"));
    expect(input.value).toBe("a@b.com");

    fireEvent.click(screen.getByTestId("send-email-button"));
    await waitFor(() =>
      expect(sendPhotoEmail).toHaveBeenCalledWith({
        to: "a@b.com",
        imageSrc: "blob:abc",
      }),
    );
  });

  it("on the mobile layout uses the native keyboard (no on-screen keyboard) and sanitizes input", () => {
    render(
      <StudioView
        layout="mobile"
        photo="data:image/png;base64,AAA"
        phase="result"
        generatedUrl="blob:abc"
        onSelect={vi.fn()}
        onNewSession={vi.fn()}
      />,
    );
    const input = screen.getByTestId("email-input") as HTMLInputElement;
    expect(input).not.toHaveAttribute("readonly");
    expect(screen.queryByTestId("email-keyboard")).not.toBeInTheDocument();

    // Regex sanitization strips characters that can't appear in an email
    // (spaces, !, # …), keeping letters/digits and @ . _ - +.
    fireEvent.change(input, { target: { value: "a b!c#@d e.com" } });
    expect(input.value).toBe("abc@de.com");
  });
});
