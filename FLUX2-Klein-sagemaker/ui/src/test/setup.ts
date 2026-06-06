// Vitest + React Testing Library setup.
// Registers jest-dom matchers (e.g. toBeInTheDocument) and clears the DOM
// between tests so component tests stay isolated.
import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
});
