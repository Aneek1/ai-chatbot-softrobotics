import { stripRemovedCitations } from "../lib/citations";
import type {
  ChatEvent,
  Citations,
  LanguagePayload,
  Notice,
  Source,
  StoredChat,
  StreamError,
  WebResult,
} from "../lib/types";

export type TurnStatus = "streaming" | "done" | "failed";

export interface Turn {
  id: string;
  question: string;
  language: LanguagePayload | null;
  sources: Source[];
  web: WebResult[];
  answer: string;
  citations: Citations | null;
  notices: Notice[];
  error: StreamError | null;
  answerLanguage: string | null;
  status: TurnStatus;
}

export interface ChatState {
  chatId: string | null;
  turns: Turn[];
}

export type ChatAction =
  | { type: "ask"; id: string; question: string }
  | { type: "event"; event: ChatEvent }
  | { type: "failed"; error: StreamError }
  | { type: "load"; chatId: string; turns: Turn[] }
  | { type: "reset" };

export const emptyChat: ChatState = { chatId: null, turns: [] };

export function newTurn(id: string, question: string): Turn {
  return {
    id,
    question,
    language: null,
    sources: [],
    web: [],
    answer: "",
    citations: null,
    notices: [],
    error: null,
    answerLanguage: null,
    status: "streaming",
  };
}

function withLast(state: ChatState, change: (turn: Turn) => Turn): ChatState {
  if (state.turns.length === 0) {
    return state;
  }
  const turns = [...state.turns];
  turns[turns.length - 1] = change(turns[turns.length - 1]);
  return { ...state, turns };
}

function applyEvent(state: ChatState, event: ChatEvent): ChatState {
  switch (event.type) {
    case "language":
      return withLast(state, (turn) => ({ ...turn, language: event.data }));
    case "sources":
      return withLast(state, (turn) => ({ ...turn, sources: event.data }));
    case "web":
      return withLast(state, (turn) => ({ ...turn, web: event.data }));
    case "token":
      return withLast(state, (turn) => ({ ...turn, answer: turn.answer + event.data.text }));
    case "citations":
      return withLast(state, (turn) => ({
        ...turn,
        citations: event.data,
        answer: stripRemovedCitations(turn.answer, event.data.removed),
      }));
    case "notice":
      return withLast(state, (turn) => ({ ...turn, notices: [...turn.notices, event.data] }));
    case "error":
      return withLast(state, (turn) => ({ ...turn, error: event.data, status: "failed" }));
    case "done": {
      const next = withLast(state, (turn) => ({
        ...turn,
        answerLanguage: event.data.answer_language,
        status: turn.status === "failed" ? "failed" : "done",
      }));
      return { ...next, chatId: event.data.chat_id };
    }
  }
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "ask":
      return { ...state, turns: [...state.turns, newTurn(action.id, action.question)] };
    case "event":
      return applyEvent(state, action.event);
    case "failed":
      return withLast(state, (turn) => ({ ...turn, error: action.error, status: "failed" }));
    case "load":
      return { chatId: action.chatId, turns: action.turns };
    case "reset":
      return emptyChat;
  }
}

/** Pairs each stored question with the answer that follows it. */
export function turnsFromChat(chat: StoredChat): Turn[] {
  const turns: Turn[] = [];
  chat.messages.forEach((message, index) => {
    if (message.role !== "user") {
      return;
    }
    const answer = chat.messages[index + 1];
    const turn = newTurn(`${chat.id}-${index}`, message.content);
    turn.language = message.language ?? null;
    if (answer !== undefined && answer.role === "assistant") {
      turn.answer = answer.content;
      turn.sources = answer.sources ?? [];
      turn.web = answer.web ?? [];
      turn.citations = answer.citations ?? null;
      turn.answerLanguage = answer.answer_language ?? null;
      turn.status = "done";
    } else {
      turn.status = "failed";
      turn.error = { code: "incomplete", message: "This question has no stored answer" };
    }
    turns.push(turn);
  });
  return turns;
}
