import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useChat } from "./useChat";
import type { ChatEvent } from "../lib/types";
import { language } from "../test/helpers";

const streamChat = vi.fn();
vi.mock("../lib/api", () => ({
  ApiError: class ApiError extends Error {},
  streamChat: (...args: unknown[]) => streamChat(...args),
}));

function events(...items: ChatEvent[]) {
  return async function* generate() {
    for (const item of items) {
      yield item;
    }
  };
}

beforeEach(() => {
  streamChat.mockReset();
});

describe("useChat", () => {
  it("streams the events of an answer into the state", async () => {
    streamChat.mockImplementation(
      events(
        { type: "language", data: language() },
        { type: "token", data: { text: "Silikon" } },
        { type: "done", data: { answer_language: "ind_Latn", chat_id: "c1" } },
      ),
    );
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask("Apa itu silikon?", "ollama", null);
    });
    await waitFor(() => expect(result.current.state.turns[0].status).toBe("done"));
    expect(result.current.state.turns[0].answer).toBe("Silikon");
    expect(result.current.state.chatId).toBe("c1");
  });

  it("sends the chat id and the override the second time", async () => {
    streamChat.mockImplementation(events({ type: "done", data: { answer_language: "eng_Latn", chat_id: "c1" } }));
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask("first", "ollama", null);
    });
    await act(async () => {
      await result.current.ask("second", "gemini", "tam_Taml");
    });
    expect(streamChat.mock.calls[1][0]).toEqual({
      message: "second",
      model: "gemini",
      chat_id: "c1",
      language_override: "tam_Taml",
    });
  });

  it("reports a stream that never starts", async () => {
    streamChat.mockImplementation(() => {
      throw new Error("connection refused");
    });
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask("hello", "ollama", null);
    });
    expect(result.current.state.turns[0].status).toBe("failed");
    expect(result.current.state.turns[0].error?.code).toBe("network_error");
  });

  it("stops an answer on request", async () => {
    streamChat.mockImplementation(async function* stall() {
      yield { type: "token", data: { text: "partial" } } as ChatEvent;
      await new Promise(() => undefined);
    });
    const { result } = renderHook(() => useChat());
    act(() => {
      void result.current.ask("hello", "ollama", null);
    });
    await waitFor(() => expect(result.current.state.turns[0].answer).toBe("partial"));
    act(() => result.current.stop());
    await waitFor(() => expect(result.current.state.turns[0].error?.code).toBe("cancelled"));
  });

  it("starts a new conversation on reset", async () => {
    streamChat.mockImplementation(events({ type: "done", data: { answer_language: "eng_Latn", chat_id: "c1" } }));
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask("hello", "ollama", null);
    });
    act(() => result.current.reset());
    expect(result.current.state.turns).toHaveLength(0);
    expect(result.current.state.chatId).toBeNull();
  });

  it("does not misattribute a second stream's events onto the first turn when both run", async () => {
    let releaseFirst: (() => void) | null = null;
    streamChat.mockImplementationOnce(async function* firstStream() {
      yield { type: "token", data: { text: "first-partial" } } as ChatEvent;
      await new Promise<void>((resolve) => {
        releaseFirst = resolve;
      });
      yield { type: "done", data: { answer_language: "eng_Latn", chat_id: "c1" } } as ChatEvent;
    });
    streamChat.mockImplementationOnce(
      events(
        { type: "token", data: { text: "second-answer" } },
        { type: "done", data: { answer_language: "eng_Latn", chat_id: "c1" } },
      ),
    );
    const { result } = renderHook(() => useChat());
    act(() => {
      void result.current.ask("first", "ollama", null);
    });
    await waitFor(() => expect(result.current.state.turns[0]?.answer).toBe("first-partial"));

    await act(async () => {
      await result.current.ask("second", "ollama", null);
    });
    expect(result.current.state.turns[1].answer).toBe("second-answer");
    expect(result.current.state.turns[0].answer).toBe("first-partial");

    await act(async () => {
      releaseFirst?.();
      await Promise.resolve();
      await Promise.resolve();
    });
    // The first stream's late "done" must not have overwritten the second turn's answer.
    expect(result.current.state.turns[1].answer).toBe("second-answer");
  });
});
