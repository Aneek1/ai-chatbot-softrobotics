import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ThemeToggle } from "./ThemeToggle";

describe("ThemeToggle", () => {
  it("names the theme it would switch to", () => {
    render(<ThemeToggle theme="light" onToggle={() => undefined} />);
    expect(screen.getByRole("button", { name: "Switch to the dark theme" })).toBeInTheDocument();
  });

  it("reports the theme in use", () => {
    render(<ThemeToggle theme="dark" onToggle={() => undefined} />);
    expect(screen.getByRole("button")).toHaveAttribute("aria-pressed", "true");
  });

  it("calls back when clicked", async () => {
    const onToggle = vi.fn();
    render(<ThemeToggle theme="light" onToggle={onToggle} />);
    await userEvent.click(screen.getByRole("button"));
    expect(onToggle).toHaveBeenCalledOnce();
  });
});
