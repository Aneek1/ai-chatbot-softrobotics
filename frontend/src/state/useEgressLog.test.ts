import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useEgressLog } from "./useEgressLog";
import type { EgressEntry } from "../lib/types";

const streamEgress = vi.fn();
vi.mock("../lib/api", () => ({ streamEgress: (...args: unknown[]) => streamEgress(...args) }));

function entry(host: string): EgressEntry {
  return { time: "2026-09-16T10:00:00.000Z", host, port: 443, verdict: "allowed", component: "ollama" };
}

describe("useEgressLog", () => {
  it("keeps the newest attempt first", async () => {
    streamEgress.mockImplementation(async function* stream() {
      yield entry("127.0.0.1");
      yield entry("html.duckduckgo.com");
    });
    const { result } = renderHook(() => useEgressLog(true));
    await waitFor(() => expect(result.current).toHaveLength(2));
    expect(result.current[0].host).toBe("html.duckduckgo.com");
  });

  it("ends the stream when the view goes away", async () => {
    let signal: AbortSignal | undefined;
    streamEgress.mockImplementation(async function* stream(given: AbortSignal) {
      signal = given;
      yield entry("127.0.0.1");
      await new Promise(() => undefined);
    });
    const { unmount } = renderHook(() => useEgressLog(true));
    await waitFor(() => expect(signal).toBeDefined());
    unmount();
    expect(signal?.aborted).toBe(true);
  });
});
