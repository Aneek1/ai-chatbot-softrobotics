import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { Inspector } from "./Inspector";

const panels = [
  { id: "language", label: "Language", content: <p>language panel</p> },
  { id: "sources", label: "Sources", content: <p>sources panel</p> },
  { id: "privacy", label: "Privacy", content: <p>privacy panel</p> },
];

describe("Inspector", () => {
  it("opens on the first tab", () => {
    render(<Inspector panels={panels} />);
    expect(screen.getByRole("tab", { name: "Language" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("language panel")).toBeInTheDocument();
  });

  it("shows the panel of the tab that was clicked", async () => {
    render(<Inspector panels={panels} />);
    await userEvent.click(screen.getByRole("tab", { name: "Sources" }));
    expect(screen.getByText("sources panel")).toBeInTheDocument();
    expect(screen.queryByText("language panel")).not.toBeInTheDocument();
  });

  it("moves to the next tab with the right arrow", async () => {
    render(<Inspector panels={panels} />);
    screen.getByRole("tab", { name: "Language" }).focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Sources" })).toHaveFocus();
    expect(screen.getByText("sources panel")).toBeInTheDocument();
  });

  it("wraps to the last tab with the left arrow", async () => {
    render(<Inspector panels={panels} />);
    screen.getByRole("tab", { name: "Language" }).focus();
    await userEvent.keyboard("{ArrowLeft}");
    expect(screen.getByRole("tab", { name: "Privacy" })).toHaveFocus();
  });

  it("keeps one tab in the tab order and connects it to its panel", () => {
    render(<Inspector panels={panels} />);
    expect(screen.getByRole("tab", { name: "Language" })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("tab", { name: "Sources" })).toHaveAttribute("tabindex", "-1");
    expect(screen.getByRole("tabpanel")).toHaveAttribute("aria-labelledby", "tab-language");
  });

  it("can be closed when it is shown as a sheet", async () => {
    const onClose = vi.fn();
    render(<Inspector panels={panels} onClose={onClose} />);
    await userEvent.click(screen.getByRole("button", { name: "Close the inspector" }));
    expect(onClose).toHaveBeenCalledOnce();
  });
});
