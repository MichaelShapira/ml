/**
 * PromptEditModal — admin-only editor for a single effect's prompt.
 *
 * Shown when an admin taps the small "Edit prompt" pencil on an effect button
 * (see {@link StudioView}). Displays the effect's current effective prompt
 * (admin override if set, else the catalog default) in an editable textarea and
 * offers two actions:
 *   - Save: persist the edited text as a custom override (used by all users).
 *   - Restore to default: reset the stored prompt to the built-in catalog
 *     default.
 *
 * Persistence goes through `api/prompts.ts` (DynamoDB, AWS SDK v3). Writes
 * require the Admin_Role; a non-admin reaching here would be denied by IAM. The
 * pencil itself is only rendered for admins, so this is a cosmetic+IAM gate.
 */
import { useState } from "react";
import { TouchButton, PrimaryButton } from "../theme";
import {
  findEffect,
  getDefaultPromptForEffect,
  getPromptForEffect,
} from "./effects";
import { savePromptOverride, restorePromptDefault } from "../api/prompts";

export interface PromptEditModalProps {
  /** The effect being edited. */
  effectId: string;
  /** Called after a successful save/restore, or on cancel. */
  onClose: () => void;
}

type Status = "idle" | "saving" | "restoring" | "error";

export function PromptEditModal({ effectId, onClose }: PromptEditModalProps) {
  const effect = findEffect(effectId);
  const [text, setText] = useState<string>(() => getPromptForEffect(effectId));
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  const busy = status === "saving" || status === "restoring";
  const isDefault = text.trim() === getDefaultPromptForEffect(effectId).trim();

  const handleSave = async () => {
    setStatus("saving");
    setError(null);
    try {
      await savePromptOverride(effectId, text);
      onClose();
    } catch (err) {
      setStatus("error");
      setError(messageOf(err, "Couldn't save the prompt. Check admin access and try again."));
    }
  };

  const handleRestore = async () => {
    setStatus("restoring");
    setError(null);
    try {
      await restorePromptDefault(effectId);
      setText(getDefaultPromptForEffect(effectId));
      onClose();
    } catch (err) {
      setStatus("error");
      setError(messageOf(err, "Couldn't restore the default. Check admin access and try again."));
    }
  };

  return (
    <div
      className="prompt-modal__backdrop"
      role="presentation"
      onClick={busy ? undefined : onClose}
    >
      <div
        className="prompt-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`Edit prompt — ${effect?.label ?? effectId}`}
        data-testid="prompt-edit-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="prompt-modal__title">Edit prompt — {effect?.label ?? effectId}</h2>
        <p className="prompt-modal__hint">
          This prompt is used for everyone. {isDefault ? "Currently the default." : "Customized."}
        </p>
        <textarea
          className="prompt-modal__textarea"
          data-testid="prompt-edit-textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={10}
          disabled={busy}
          spellCheck={false}
        />
        {error && (
          <p className="kiosk-error" role="alert" data-testid="prompt-edit-error">
            {error}
          </p>
        )}
        <div className="prompt-modal__actions">
          <TouchButton
            variant="secondary"
            testId="prompt-edit-restore"
            disabled={busy || isDefault}
            onClick={() => void handleRestore()}
          >
            {status === "restoring" ? "Restoring…" : "Restore to default"}
          </TouchButton>
          <div className="prompt-modal__actions-right">
            <TouchButton
              variant="secondary"
              testId="prompt-edit-cancel"
              disabled={busy}
              onClick={onClose}
            >
              Cancel
            </TouchButton>
            <PrimaryButton
              testId="prompt-edit-save"
              disabled={busy || text.trim().length === 0}
              onClick={() => void handleSave()}
            >
              {status === "saving" ? "Saving…" : "Save"}
            </PrimaryButton>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Extract a user-facing message from an unknown thrown value. */
function messageOf(err: unknown, fallback: string): string {
  if (err instanceof Error && err.message) {
    return err.message;
  }
  return fallback;
}

export default PromptEditModal;
