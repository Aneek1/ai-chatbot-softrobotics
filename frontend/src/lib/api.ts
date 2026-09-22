import { readSse } from "./sse";
import type {
  ChatEvent,
  ChatSummary,
  EgressEntry,
  Health,
  Mode,
  ModelName,
  StoredChat,
} from "./types";

/** The backend's parsed `{code, message}` error shape, when the body is JSON in that shape. */
export interface ApiErrorBody {
  code?: string;
  message?: string;
}

export class ApiError extends Error {
  readonly status: number;
  /**
   * The response body, when one was readable: a parsed `{code, message}`
   * object when the body was JSON, the raw text otherwise, or null when
   * there was no body or it could not be read at all.
   */
  readonly body: ApiErrorBody | string | null;

  constructor(status: number, message: string, body: ApiErrorBody | string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  return typeof value === "object" && value !== null;
}

/**
 * Reads a failed response's body so the caller can see why it failed.
 * Prefers a parsed `{code, message}` object when the body is JSON, falls
 * back to the raw text otherwise, and never throws: a body that cannot be
 * read or parsed must not mask the original HTTP failure.
 */
async function readErrorBody(response: Response): Promise<ApiErrorBody | string | null> {
  let text: string;
  try {
    text = await response.text();
  } catch {
    return null;
  }
  if (text === "") {
    return null;
  }
  try {
    const parsed: unknown = JSON.parse(text);
    if (isApiErrorBody(parsed)) {
      return parsed;
    }
  } catch {
    // Not JSON — fall through to the raw text.
  }
  return text;
}

function describeErrorBody(body: ApiErrorBody | string | null): string | null {
  if (body === null) {
    return null;
  }
  return typeof body === "string" ? body : (body.message ?? body.code ?? null);
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const body = await readErrorBody(response);
    const detail = describeErrorBody(body);
    const summary = `${init.method ?? "GET"} ${path} failed with ${response.status}`;
    throw new ApiError(response.status, detail ? `${summary}: ${detail}` : summary, body);
  }
  return response;
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await request(path, init);
  return (await response.json()) as T;
}

export function getHealth(): Promise<Health> {
  return requestJson<Health>("/api/health");
}

export function getMode(): Promise<Mode> {
  return requestJson<Mode>("/api/mode");
}

export function setMode(isPrivate: boolean): Promise<Mode> {
  return requestJson<Mode>("/api/mode", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ private: isPrivate }),
  });
}

export function listChats(): Promise<ChatSummary[]> {
  return requestJson<ChatSummary[]>("/api/chats");
}

export function getChat(chatId: string): Promise<StoredChat> {
  return requestJson<StoredChat>(`/api/chats/${encodeURIComponent(chatId)}`);
}

export async function deleteChat(chatId: string): Promise<void> {
  await request(`/api/chats/${encodeURIComponent(chatId)}`, { method: "DELETE" });
}

export interface ChatRequest {
  message: string;
  model: ModelName;
  chat_id?: string;
  language_override?: string;
}

const CHAT_EVENT_TAGS = new Set<ChatEvent["type"]>([
  "language",
  "sources",
  "web",
  "token",
  "citations",
  "notice",
  "error",
  "done",
]);

/**
 * Turns a raw SSE message into a typed ChatEvent, or null when it should be
 * dropped. `JSON.parse(...) as ChatEvent` trusts backend schema drift
 * blindly: a malformed or unrecognised event would otherwise be handed to
 * components that assume the type is correct. This only checks the shape
 * cheaply (an object with a known event tag) rather than validating each
 * event's payload fully. An unrecognised or malformed event is dropped with
 * a console warning instead of thrown, because one bad event should not
 * abort an otherwise-good stream.
 */
function parseChatEvent(rawType: string, rawData: string): ChatEvent | null {
  if (!CHAT_EVENT_TAGS.has(rawType as ChatEvent["type"])) {
    console.warn(`streamChat: dropping an event with an unrecognised type: ${rawType}`);
    return null;
  }
  let data: unknown;
  try {
    data = JSON.parse(rawData);
  } catch {
    console.warn(`streamChat: dropping a "${rawType}" event with unparsable JSON data`);
    return null;
  }
  if (typeof data !== "object" || data === null) {
    console.warn(`streamChat: dropping a "${rawType}" event whose data was not an object`);
    return null;
  }
  return { type: rawType, data } as ChatEvent;
}

export async function* streamChat(body: ChatRequest, signal?: AbortSignal): AsyncGenerator<ChatEvent> {
  const response = await request("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (response.body === null) {
    throw new ApiError(response.status, "POST /api/chat returned no body");
  }
  for await (const message of readSse(response.body)) {
    const event = parseChatEvent(message.event, message.data);
    if (event !== null) {
      yield event;
    }
  }
}

export async function* streamEgress(signal?: AbortSignal): AsyncGenerator<EgressEntry> {
  const response = await request("/api/egress", {
    headers: { Accept: "text/event-stream" },
    signal,
  });
  if (response.body === null) {
    throw new ApiError(response.status, "GET /api/egress returned no body");
  }
  for await (const message of readSse(response.body)) {
    if (message.event === "connection") {
      yield JSON.parse(message.data) as EgressEntry;
    }
  }
}
