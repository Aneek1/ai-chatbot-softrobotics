export type Stage = "general" | "specialist" | "rule";
export type ModelName = "ollama" | "gemini";

export interface Candidate {
  code: string;
  name: string;
  script: string;
  probability: number;
}

export interface LanguagePayload {
  candidates: Candidate[];
  chosen: string;
  chosen_name: string;
  uncertain: boolean;
  stage: Stage;
}

export interface Source {
  id: string;
  title: string;
  snippet: string;
  language: string;
  source: string;
  url: string;
  licence: string;
}

export interface WebResult {
  title: string;
  url: string;
  snippet: string;
  engine: string;
}

export interface Notice {
  kind: string;
  message: string;
}

export interface StreamError {
  code: string;
  message: string;
}

export interface Citations {
  valid: number[];
  removed: number[];
}

export interface DonePayload {
  answer_language: string;
  chat_id: string;
}

export type ChatEvent =
  | { type: "language"; data: LanguagePayload }
  | { type: "sources"; data: Source[] }
  | { type: "web"; data: WebResult[] }
  | { type: "token"; data: { text: string } }
  | { type: "citations"; data: Citations }
  | { type: "notice"; data: Notice }
  | { type: "error"; data: StreamError }
  | { type: "done"; data: DonePayload };

export interface Mode {
  private: boolean;
  proxy_configured: boolean;
  ollama_reachable: boolean;
}

export interface Health {
  detector: string;
  model_files: Record<string, boolean>;
  ollama_reachable: boolean;
  gemini_configured: boolean;
  web_search: { normal: string | null; private: string | null };
  private: boolean;
  egress_guard: boolean;
  documents_by_language: Record<string, number>;
}

export interface ChatSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface StoredMessage {
  role: "user" | "assistant";
  content: string;
  created_at: string;
  language?: LanguagePayload;
  answer_language?: string;
  sources?: Source[];
  web?: WebResult[];
  citations?: Citations;
}

export interface StoredChat extends ChatSummary {
  messages: StoredMessage[];
}

export interface EgressEntry {
  time: string;
  host: string;
  port: number | null;
  verdict: "allowed" | "blocked";
  component: string;
}
