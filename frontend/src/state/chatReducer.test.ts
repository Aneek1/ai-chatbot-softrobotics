import { describe, expect, it } from "vitest";

import { chatReducer, emptyChat, turnsFromChat } from "./chatReducer";
import type { ChatState } from "./chatReducer";
import { language, source } from "../test/helpers";

function asked(question = "Apa itu silikon?"): ChatState {
  return chatReducer(emptyChat, { type: "ask", id: "turn-1", question });
}

describe("chatReducer", () => {
  it("adds a streaming turn for the question", () => {
    const state = asked();
    expect(state.turns).toHaveLength(1);
    expect(state.turns[0].question).toBe("Apa itu silikon?");
    expect(state.turns[0].status).toBe("streaming");
  });

  it("stores the detected language on the newest turn", () => {
    const state = chatReducer(asked(), { type: "event", event: { type: "language", data: language() } });
    expect(state.turns[0].language?.chosen).toBe("ind_Latn");
  });

  it("appends the streamed tokens", () => {
    let state = chatReducer(asked(), { type: "event", event: { type: "token", data: { text: "Silikon " } } });
    state = chatReducer(state, { type: "event", event: { type: "token", data: { text: "adalah" } } });
    expect(state.turns[0].answer).toBe("Silikon adalah");
  });

  it("removes the citation markers the backend rejected", () => {
    let state = chatReducer(asked(), {
      type: "event",
      event: { type: "token", data: { text: "Dituang [1]. Mengeras [7]." } },
    });
    state = chatReducer(state, {
      type: "event",
      event: { type: "citations", data: { valid: [1], removed: [7] } },
    });
    expect(state.turns[0].answer).toBe("Dituang [1]. Mengeras.");
    expect(state.turns[0].citations?.removed).toEqual([7]);
  });

  it("stores the retrieved sources and the web snippets", () => {
    let state = chatReducer(asked(), { type: "event", event: { type: "sources", data: [source()] } });
    state = chatReducer(state, {
      type: "event",
      event: {
        type: "web",
        data: [{ title: "Casting guide", url: "https://example.invalid/g", snippet: "Degas", engine: "duckduckgo" }],
      },
    });
    expect(state.turns[0].sources).toHaveLength(1);
    expect(state.turns[0].web[0].engine).toBe("duckduckgo");
  });

  it("keeps the notices in the order they arrived", () => {
    let state = chatReducer(asked(), {
      type: "event",
      event: { type: "notice", data: { kind: "web_search_off", message: "Web search is off" } },
    });
    state = chatReducer(state, {
      type: "event",
      event: { type: "notice", data: { kind: "no_sources", message: "No sources found" } },
    });
    expect(state.turns[0].notices.map((notice) => notice.kind)).toEqual(["web_search_off", "no_sources"]);
  });

  it("marks a failed turn and keeps what was streamed", () => {
    let state = chatReducer(asked(), { type: "event", event: { type: "token", data: { text: "Sili" } } });
    state = chatReducer(state, {
      type: "event",
      event: { type: "error", data: { code: "provider_timeout", message: "Ollama timed out" } },
    });
    expect(state.turns[0].status).toBe("failed");
    expect(state.turns[0].error?.code).toBe("provider_timeout");
    expect(state.turns[0].answer).toBe("Sili");
  });

  it("records the answer language and the chat id when the stream ends", () => {
    const state = chatReducer(asked(), {
      type: "event",
      event: { type: "done", data: { answer_language: "ind_Latn", chat_id: "c1" } },
    });
    expect(state.turns[0].status).toBe("done");
    expect(state.turns[0].answerLanguage).toBe("ind_Latn");
    expect(state.chatId).toBe("c1");
  });

  it("ignores an event that arrives with no turn to attach it to", () => {
    const state = chatReducer(emptyChat, { type: "event", event: { type: "token", data: { text: "x" } } });
    expect(state).toEqual(emptyChat);
  });

  it("marks a turn failed from outside the stream", () => {
    const state = chatReducer(asked(), {
      type: "failed",
      error: { code: "network_error", message: "The backend could not be reached" },
    });
    expect(state.turns[0].status).toBe("failed");
  });

  it("clears everything on reset", () => {
    const state = chatReducer(
      chatReducer(asked(), { type: "event", event: { type: "done", data: { answer_language: "ind_Latn", chat_id: "c1" } } }),
      { type: "reset" },
    );
    expect(state).toEqual(emptyChat);
  });

  it("rebuilds turns from a stored chat", () => {
    const turns = turnsFromChat({
      id: "c1",
      title: "Apa itu silikon?",
      created_at: "2026-09-16T10:00:00.000Z",
      updated_at: "2026-09-16T10:00:01.000Z",
      messages: [
        { role: "user", content: "Apa itu silikon?", created_at: "2026-09-16T10:00:00.000Z", language: language() },
        {
          role: "assistant",
          content: "Silikon adalah [1].",
          created_at: "2026-09-16T10:00:01.000Z",
          answer_language: "ind_Latn",
          sources: [source()],
          web: [],
          citations: { valid: [1], removed: [] },
        },
        { role: "user", content: "Berapa lama mengeras?", created_at: "2026-09-16T10:01:00.000Z", language: language() },
      ],
    });
    expect(turns).toHaveLength(2);
    expect(turns[0].answer).toBe("Silikon adalah [1].");
    expect(turns[0].status).toBe("done");
    expect(turns[1].answer).toBe("");
  });
});
