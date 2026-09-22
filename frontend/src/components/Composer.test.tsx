import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { Composer } from "./Composer";

function setup(overrides: Partial<Parameters<typeof Composer>[0]> = {}) {
  const props = {
    onSend: vi.fn(),
    onStop: vi.fn(),
    streaming: false,
    disabled: false,
    ...overrides,
  };
  render(<Composer {...props} />);
  return props;
}

describe("Composer", () => {
  it("sends the trimmed question and clears the field", async () => {
    const { onSend } = setup();
    const field = screen.getByLabelText("Your question");
    await userEvent.type(field, "  Apa itu silikon?  ");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).toHaveBeenCalledWith("Apa itu silikon?");
    expect(field).toHaveValue("");
  });

  it("sends on Enter", async () => {
    const { onSend } = setup();
    await userEvent.type(screen.getByLabelText("Your question"), "소프트 로봇{Enter}");
    expect(onSend).toHaveBeenCalledWith("소프트 로봇");
  });

  it("starts a new line on Shift+Enter", async () => {
    const { onSend } = setup();
    const field = screen.getByLabelText("Your question");
    await userEvent.type(field, "one{Shift>}{Enter}{/Shift}two");
    expect(onSend).not.toHaveBeenCalled();
    expect(field).toHaveValue("one\ntwo");
  });

  it("does not send an empty question", async () => {
    const { onSend } = setup();
    await userEvent.type(screen.getByLabelText("Your question"), "   {Enter}");
    expect(onSend).not.toHaveBeenCalled();
  });

  it("disables sending while the app is not ready", () => {
    setup({ disabled: true });
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("offers Stop while an answer is streaming", async () => {
    const { onStop } = setup({ streaming: true });
    await userEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(onStop).toHaveBeenCalledOnce();
  });
});
