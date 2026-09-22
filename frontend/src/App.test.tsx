import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { ChatEvent } from "./lib/types";
import { health, language, mode, source, stubMatchMedia } from "./test/helpers";

const api = vi.hoisted(() => ({
  getHealth: vi.fn(),
  getMode: vi.fn(),
  setMode: vi.fn(),
  listChats: vi.fn(),
  getChat: vi.fn(),
  deleteChat: vi.fn(),
  streamChat: vi.fn(),
  streamEgress: vi.fn(),
}));

vi.mock("./lib/api", () => ({
  ApiError: class ApiError extends Error {},
  getHealth: api.getHealth,
  getMode: api.getMode,
  setMode: api.setMode,
  listChats: api.listChats,
  getChat: api.getChat,
  deleteChat: api.deleteChat,
  streamChat: api.streamChat,
  streamEgress: api.streamEgress,
}));

function events(...items: ChatEvent[]) {
  return async function* generate() {
    for (const item of items) {
      yield item;
    }
  };
}

const answer = events(
  { type: "language", data: language() },
  { type: "sources", data: [source()] },
  { type: "token", data: { text: "Silikon adalah polimer [1]." } },
  { type: "citations", data: { valid: [1], removed: [] } },
  { type: "done", data: { answer_language: "ind_Latn", chat_id: "c1" } },
);

beforeEach(() => {
  stubMatchMedia(false);
  api.getHealth.mockResolvedValue(health());
  api.getMode.mockResolvedValue(mode());
  api.listChats.mockResolvedValue([
    { id: "c1", title: "Apa itu silikon?", created_at: "2026-09-16T08:00:00.000Z", updated_at: "2026-09-16T08:10:00.000Z" },
  ]);
  api.streamEgress.mockImplementation(async function* stream() {});
  api.streamChat.mockImplementation(answer);
});

async function ask(question: string) {
  await userEvent.type(screen.getByLabelText("Your question"), question);
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
}

describe("App", () => {
  it("loads the health, the mode and the chat list", async () => {
    render(<App />);
    expect(await screen.findByText("Detector: glotlid-q.ftz")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open chat: Apa itu silikon?" })).toBeInTheDocument();
  });

  it("streams an answer into the transcript", async () => {
    render(<App />);
    await ask("Apa itu silikon?");
    expect(await screen.findByText("Silikon adalah polimer [1].")).toBeInTheDocument();
    expect(api.streamChat.mock.calls[0][0].message).toBe("Apa itu silikon?");
  });

  it("shows the detected language for the newest answer", async () => {
    render(<App />);
    await ask("Apa itu silikon?");
    const detected = within(await screen.findByLabelText("Detected language"));
    expect(detected.getByText("Indonesian")).toBeInTheDocument();
    expect(detected.getByText("ind_Latn")).toBeInTheDocument();
  });

  it("explains an error the backend reported", async () => {
    api.streamChat.mockImplementation(
      events({ type: "error", data: { code: "provider_timeout", message: "Ollama timed out" } }),
    );
    render(<App />);
    await ask("Apa itu silikon?");
    expect(await screen.findByRole("alert")).toHaveTextContent("Ollama timed out");
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("switches private mode from the header", async () => {
    api.setMode.mockResolvedValue(mode({ private: true }));
    render(<App />);
    await userEvent.click((await screen.findAllByRole("switch", { name: "Private mode" }))[0]);
    await waitFor(() => expect(api.setMode).toHaveBeenCalledWith(true));
    expect(await screen.findByText("Private")).toBeInTheDocument();
  });

  it("opens the inspector as a sheet on a narrow screen", async () => {
    render(<App />);
    const inspector = await screen.findByTestId("inspector-region");
    expect(inspector).toHaveAttribute("data-open", "false");
    await userEvent.click(screen.getByRole("button", { name: "Details" }));
    expect(screen.getByTestId("inspector-region")).toHaveAttribute("data-open", "true");
  });
});
