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
  | { type: "event"; turnId: string; event: ChatEvent }
  | { type: "failed"; turnId: string; error: StreamError }
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

/**
 * Applies `change` to the turn identified by `turnId`, if it still exists in
 * `state`. A turn stops existing once a second `ask` supersedes it as the
 * event target for a caller still holding a stale id, or once `load` swaps
 * in a whole new set of turns (e.g. the user switched chats mid-stream). In
 * either case the event is for a turn that is no longer part of this state,
 * so it is dropped silently rather than corrupting whatever turn happens to
 * be last, or throwing.
 */
function withTurn(state: ChatState, turnId: string, change: (turn: Turn) => Turn): ChatState {
  const index = state.turns.findIndex((turn) => turn.id === turnId);
  if (index === -1) {
    return state;
  }
  const turns = [...state.turns];
  turns[index] = change(turns[index]);
  return { ...state, turns };
}

function applyEvent(state: ChatState, turnId: string, event: ChatEvent): ChatState {
  switch (event.type) {
    case "language":
      return withTurn(state, turnId, (turn) => ({ ...turn, language: event.data }));
    case "sources":
      return withTurn(state, turnId, (turn) => ({ ...turn, sources: event.data }));
    case "web":
      return withTurn(state, turnId, (turn) => ({ ...turn, web: event.data }));
    case "token":
      // Applied as-is: two genuinely distinct token events can legitimately
      // carry identical text, so the reducer cannot tell a real repeat from
      // a redelivered one. Guaranteeing at-most-once delivery is the SSE
      // transport's job (e.g. de-duplicating by event id), not this
      // reducer's.
      return withTurn(state, turnId, (turn) => ({ ...turn, answer: turn.answer + event.data.text }));
    case "citations":
      return withTurn(state, turnId, (turn) => ({
        ...turn,
        citations: event.data,
        answer: stripRemovedCitations(turn.answer, event.data.removed),
      }));
    case "notice":
      return withTurn(state, turnId, (turn) => ({ ...turn, notices: [...turn.notices, event.data] }));
    case "error":
      return withTurn(state, turnId, (turn) => ({ ...turn, error: event.data, status: "failed" }));
    case "done": {
      if (!state.turns.some((turn) => turn.id === turnId)) {
        return state;
      }
      const next = withTurn(state, turnId, (turn) => ({
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
      return applyEvent(state, action.turnId, action.event);
    case "failed":
      return withTurn(state, action.turnId, (turn) => ({ ...turn, error: action.error, status: "failed" }));
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
