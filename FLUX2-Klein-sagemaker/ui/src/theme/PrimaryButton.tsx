/**
 * PrimaryButton — the single, prominent primary action for a step
 * (Requirement 25.3).
 *
 * A thin convenience wrapper over {@link TouchButton} that fixes the
 * `primary` variant and defaults to the full-width `block` hero sizing, since
 * most capture-flow steps (Start, Take Photo, Continue, New Session) center a
 * single primary action in the footer. `block` can be overridden for the rare
 * step that needs an inline primary control.
 */
import type { TouchButtonProps } from "./TouchButton";
import { TouchButton } from "./TouchButton";

export type PrimaryButtonProps = Omit<TouchButtonProps, "variant">;

export function PrimaryButton({ block = true, ...rest }: PrimaryButtonProps) {
  return <TouchButton variant="primary" block={block} {...rest} />;
}

export default PrimaryButton;
