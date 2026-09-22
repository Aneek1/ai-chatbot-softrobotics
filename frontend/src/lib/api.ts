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

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(path, init);
  if (!response.ok) {
    throw new ApiError(response.status, `${init.method ?? "GET"} ${path} failed with ${response.status}`);
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
    yield { type: message.event, data: JSON.parse(message.data) } as ChatEvent;
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
