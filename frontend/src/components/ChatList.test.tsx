import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ChatList } from "./ChatList";
import type { ChatSummary } from "../lib/types";

const chats: ChatSummary[] = [
  { id: "c2", title: "소프트 로봇은 무엇인가요?", created_at: "2026-09-16T09:00:00.000Z", updated_at: "2026-09-16T09:30:00.000Z" },
  { id: "c1", title: "Apa itu silikon?", created_at: "2026-09-16T08:00:00.000Z", updated_at: "2026-09-16T08:10:00.000Z" },
];

function setup(overrides: Partial<Parameters<typeof ChatList>[0]> = {}) {
  const props = {
    chats,
    currentId: null as string | null,
    isPrivate: false,
    onNew: vi.fn(),
    onSelect: vi.fn(),
    onDelete: vi.fn(),
    ...overrides,
  };
  render(<ChatList {...props} />);
  return props;
}

describe("ChatList", () => {
  it("lists the chats in the order the backend returned them", () => {
    setup();
    const titles = screen.getAllByRole("button", { name: /Open chat/ }).map((button) => button.textContent);
    expect(titles).toEqual(["소프트 로봇은 무엇인가요?", "Apa itu silikon?"]);
  });

  it("starts a new chat", async () => {
    const { onNew } = setup();
    await userEvent.click(screen.getByRole("button", { name: "New chat" }));
    expect(onNew).toHaveBeenCalledOnce();
  });

  it("opens a chat", async () => {
    const { onSelect } = setup();
    await userEvent.click(screen.getByRole("button", { name: "Open chat: Apa itu silikon?" }));
    expect(onSelect).toHaveBeenCalledWith("c1");
  });

  it("deletes a chat", async () => {
    const { onDelete } = setup();
    await userEvent.click(screen.getByRole("button", { name: "Delete chat: Apa itu silikon?" }));
    expect(onDelete).toHaveBeenCalledWith("c1");
  });

  it("marks the chat that is open", () => {
    setup({ currentId: "c1" });
    expect(screen.getByRole("button", { name: "Open chat: Apa itu silikon?" })).toHaveAttribute(
      "aria-current",
      "true",
    );
  });

  it("says where private chats are kept when there are none", () => {
    setup({ chats: [], isPrivate: true });
    expect(
      screen.getByText("Private chats stay in memory and are cleared when private mode ends."),
    ).toBeInTheDocument();
  });
});
