import { describe, expect, it } from "vitest";

import { SseParser, readSse } from "./sse";

function stream(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}

describe("SseParser", () => {
  it("reads one event with its name and data", () => {
    const messages = new SseParser().feed('event: token\ndata: {"text": "silicone"}\n\n');
    expect(messages).toEqual([{ event: "token", data: '{"text": "silicone"}', id: null }]);
  });

  it("joins several data lines with newlines", () => {
    const messages = new SseParser().feed("event: notice\ndata: first\ndata: second\n\n");
    expect(messages[0].data).toBe("first\nsecond");
  });

  it("waits for an event split across chunks", () => {
    const parser = new SseParser();
    expect(parser.feed("event: token\ndata: {")).toEqual([]);
    const messages = parser.feed('"text": "a"}\n\n');
    expect(messages).toHaveLength(1);
    expect(messages[0].data).toBe('{"text": "a"}');
  });

  it("ignores comments and keep-alive blocks", () => {
    const parser = new SseParser();
    expect(parser.feed(": keep alive\n\n")).toEqual([]);
    expect(parser.feed("event: done\ndata: {}\n\n")).toHaveLength(1);
  });

  it("accepts carriage returns", () => {
    const messages = new SseParser().feed("event: token\r\ndata: a\r\n\r\n");
    expect(messages[0]).toEqual({ event: "token", data: "a", id: null });
  });

  it("keeps the event id", () => {
    const messages = new SseParser().feed("id: 7\nevent: connection\ndata: {}\n\n");
    expect(messages[0].id).toBe("7");
  });

  it("reads a stream in order", async () => {
    const seen: string[] = [];
    for await (const message of readSse(stream("event: a\ndata: 1\n\nevent: b\n", "data: 2\n\n"))) {
      seen.push(`${message.event}:${message.data}`);
    }
    expect(seen).toEqual(["a:1", "b:2"]);
  });
});
