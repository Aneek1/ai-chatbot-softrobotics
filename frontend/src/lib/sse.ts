export interface SseMessage {
  event: string;
  data: string;
  id: string | null;
}

function parseBlock(block: string): SseMessage | null {
  let event = "message";
  let id: string | null = null;
  const data: string[] = [];
  for (const rawLine of block.split("\n")) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line === "" || line.startsWith(":")) {
      continue;
    }
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }
    if (field === "event") {
      event = value;
    } else if (field === "data") {
      data.push(value);
    } else if (field === "id") {
      id = value;
    }
  }
  return data.length === 0 ? null : { event, data: data.join("\n"), id };
}

export class SseParser {
  private buffer = "";

  feed(chunk: string): SseMessage[] {
    this.buffer += chunk;
    const messages: SseMessage[] = [];
    for (;;) {
      const end = this.buffer.indexOf("\n\n");
      const endCrlf = this.buffer.indexOf("\r\n\r\n");
      const cut = endCrlf !== -1 && (end === -1 || endCrlf < end) ? endCrlf : end;
      if (cut === -1) {
        return messages;
      }
      const width = cut === endCrlf ? 4 : 2;
      const block = this.buffer.slice(0, cut);
      this.buffer = this.buffer.slice(cut + width);
      const message = parseBlock(block);
      if (message !== null) {
        messages.push(message);
      }
    }
  }

  /**
   * Parses whatever is left in the buffer as a final, unterminated event and
   * clears it. Call this once the underlying stream reports `done`: a
   * connection can close right after a complete event's fields but before
   * its trailing blank line (a truncating proxy, a dropped connection), and
   * without this the event is lost silently.
   */
  flush(): SseMessage | null {
    const block = this.buffer;
    this.buffer = "";
    if (block === "") {
      return null;
    }
    return parseBlock(block);
  }
}

export async function* readSse(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseMessage> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        const trailing = parser.flush();
        if (trailing !== null) {
          yield trailing;
        }
        return;
      }
      for (const message of parser.feed(decoder.decode(value, { stream: true }))) {
        yield message;
      }
    }
  } finally {
    // Cancelling releases the network connection when the caller stops early.
    await reader.cancel().catch(() => undefined);
  }
}
