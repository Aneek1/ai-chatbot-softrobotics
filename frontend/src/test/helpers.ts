import { vi } from "vitest";

import type { Health, LanguagePayload, Mode, Source } from "../lib/types";

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export function sseResponse(text: string, status = 200): Response {
  return new Response(text, { status, headers: { "Content-Type": "text/event-stream" } });
}

/** Replaces fetch with a queue of responses and records the calls. */
export function stubFetch(responses: Response[]): ReturnType<typeof vi.fn> {
  const queue = [...responses];
  const fetchMock = vi.fn(async () => {
    const next = queue.shift();
    if (next === undefined) {
      throw new Error("fetch was called more times than the test provided responses");
    }
    return next;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

export function stubMatchMedia(dark: boolean): void {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: dark && query.includes("dark"),
    media: query,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  }));
}

export function language(overrides: Partial<LanguagePayload> = {}): LanguagePayload {
  return {
    candidates: [
      { code: "ind_Latn", name: "Indonesian", script: "Latn", probability: 0.62 },
      { code: "zsm_Latn", name: "Malay", script: "Latn", probability: 0.31 },
      { code: "eng_Latn", name: "English", script: "Latn", probability: 0.04 },
    ],
    chosen: "ind_Latn",
    chosen_name: "Indonesian",
    uncertain: false,
    stage: "general",
    ...overrides,
  };
}

export function source(overrides: Partial<Source> = {}): Source {
  return {
    id: "arxiv:2401.00001:0",
    title: "Soft pneumatic actuators cast in silicone",
    snippet: "Silicone is degassed and cast into a printed mould.",
    language: "eng_Latn",
    source: "arxiv",
    url: "https://arxiv.org/abs/2401.00001",
    licence: "arXiv metadata, CC0 1.0",
    ...overrides,
  };
}

export function health(overrides: Partial<Health> = {}): Health {
  return {
    detector: "glotlid-q.ftz",
    model_files: { glotlid: true, e5: true },
    ollama_reachable: true,
    gemini_configured: false,
    web_search: { normal: null, private: "duckduckgo" },
    private: false,
    egress_guard: true,
    documents_by_language: { eng_Latn: 212, ind_Latn: 9 },
    ...overrides,
  };
}

export function mode(overrides: Partial<Mode> = {}): Mode {
  return { private: false, proxy_configured: false, ollama_reachable: true, ...overrides };
}
