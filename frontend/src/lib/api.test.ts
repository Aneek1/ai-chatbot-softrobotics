import { describe, expect, it } from "vitest";

import { ApiError, deleteChat, getChat, getHealth, setMode, streamChat, streamEgress } from "./api";
import { health, jsonResponse, sseResponse, stubFetch } from "../test/helpers";

describe("api client", () => {
  it("reads health", async () => {
    stubFetch([jsonResponse(health())]);
    expect((await getHealth()).detector).toBe("glotlid-q.ftz");
  });

  it("sends the mode switch as JSON", async () => {
    const fetchMock = stubFetch([
      jsonResponse({ private: true, proxy_configured: false, ollama_reachable: true }),
    ]);
    const result = await setMode(true);
    expect(result.private).toBe(true);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/mode");
    expect(init.method).toBe("PUT");
    expect(init.body).toBe('{"private":true}');
  });

  it("turns a failed response into an ApiError carrying the status", async () => {
    stubFetch([jsonResponse({ detail: "busy" }, 409)]);
    await expect(setMode(true)).rejects.toMatchObject({ name: "ApiError", status: 409 });
    expect(new ApiError(404, "gone").message).toBe("gone");
  });

  it("carries a parsed JSON error body on a failed response", async () => {
    stubFetch([jsonResponse({ code: "mode_locked", message: "Mode is locked by the admin" }, 409)]);
    await expect(setMode(true)).rejects.toMatchObject({
      name: "ApiError",
      status: 409,
      body: { code: "mode_locked", message: "Mode is locked by the admin" },
    });
  });

  it("does not throw a second error when a failed response body is not JSON", async () => {
    stubFetch([new Response("Internal Server Error", { status: 500 })]);
    await expect(setMode(true)).rejects.toMatchObject({ name: "ApiError", status: 500, body: "Internal Server Error" });
  });

  it("escapes the chat id in the path", async () => {
    const fetchMock = stubFetch([jsonResponse({ id: "a b", title: "", created_at: "", updated_at: "", messages: [] })]);
    await getChat("a b");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/chats/a%20b");
  });

  it("deletes a chat", async () => {
    const fetchMock = stubFetch([new Response(null, { status: 204 })]);
    await deleteChat("c1");
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe("DELETE");
  });

  it("yields the chat events in order", async () => {
    stubFetch([
      sseResponse(
        'event: language\ndata: {"chosen": "ind_Latn"}\n\n' +
          'event: token\ndata: {"text": "Silikon "}\n\n' +
          'event: done\ndata: {"answer_language": "ind_Latn", "chat_id": "c1"}\n\n',
      ),
    ]);
    const seen: string[] = [];
    for await (const event of streamChat({ message: "Apa itu silikon?", model: "ollama" })) {
      seen.push(event.type);
    }
    expect(seen).toEqual(["language", "token", "done"]);
  });

  it("reports a failed chat request before any event", async () => {
    stubFetch([jsonResponse({ detail: "no" }, 500)]);
    const iterator = streamChat({ message: "hi", model: "ollama" });
    await expect(iterator.next()).rejects.toMatchObject({ status: 500 });
  });

  it("drops a chat event whose payload does not look like a chat event, without throwing", async () => {
    stubFetch([
      sseResponse(
        'event: token\ndata: "just a string, not an object"\n\n' +
          'event: token\ndata: {"text": "ok"}\n\n' +
          'event: done\ndata: {"answer_language": "ind_Latn", "chat_id": "c1"}\n\n',
      ),
    ]);
    const seen: string[] = [];
    for await (const event of streamChat({ message: "hi", model: "ollama" })) {
      seen.push(event.type);
    }
    expect(seen).toEqual(["token", "done"]);
  });

  it("yields only connection entries from the egress stream", async () => {
    stubFetch([
      sseResponse(
        ': open\n\nid: 1\nevent: connection\ndata: {"host": "localhost", "verdict": "allowed"}\n\n' +
          'event: notice\ndata: {"message": "reconnecting"}\n\n',
      ),
    ]);
    const entries = [];
    for await (const entry of streamEgress()) {
      entries.push(entry);
    }
    expect(entries).toEqual([{ host: "localhost", verdict: "allowed" }]);
  });
});
