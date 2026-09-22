import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AppHeader } from "./AppHeader";
import { health, mode } from "../test/helpers";

function setup(overrides: Partial<Parameters<typeof AppHeader>[0]> = {}) {
  const props = {
    health: health(),
    healthError: false,
    mode: mode(),
    model: "ollama" as const,
    busy: false,
    theme: "light" as const,
    onModelChange: vi.fn(),
    onPrivateChange: vi.fn(),
    onToggleTheme: vi.fn(),
    ...overrides,
  };
  render(<AppHeader {...props} />);
  return props;
}

describe("AppHeader", () => {
  it("names the app and the detector in use", () => {
    setup();
    expect(screen.getByRole("heading", { name: "Soft-robotics assistant" })).toBeInTheDocument();
    expect(screen.getByText("Detector: glotlid-q.ftz")).toBeInTheDocument();
  });

  it("shows the detector as unreachable, not unknown, when health failed to load", () => {
    setup({ health: null, healthError: true });
    expect(screen.getByText("Detector: unreachable")).toBeInTheDocument();
  });

  it("offers Gemini when it is configured", () => {
    setup({ health: health({ gemini_configured: true }) });
    expect(screen.getAllByRole("option").map((option) => option.textContent)).toEqual(["Ollama", "Gemini"]);
  });

  it("leaves Gemini out when it is not configured", () => {
    setup();
    expect(screen.getAllByRole("option").map((option) => option.textContent)).toEqual(["Ollama"]);
  });

  it("fixes the model to Ollama in private mode", () => {
    setup({ mode: mode({ private: true }), health: health({ gemini_configured: true }) });
    const select = screen.getByLabelText("Answer model");
    expect(select).toBeDisabled();
    expect(select).toHaveValue("ollama");
  });

  it("shows that private mode is on", () => {
    setup({ mode: mode({ private: true }) });
    expect(screen.getByRole("switch", { name: "Private mode" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText("Private")).toBeInTheDocument();
  });

  it("switches the theme", async () => {
    const { onToggleTheme } = setup();
    await userEvent.click(screen.getByRole("button", { name: "Switch to the dark theme" }));
    expect(onToggleTheme).toHaveBeenCalledOnce();
  });
});
