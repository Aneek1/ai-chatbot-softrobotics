import { useCallback, useEffect, useReducer, useRef } from "react";

import { streamChat } from "../lib/api";
import type { ModelName } from "../lib/types";
import { chatReducer, emptyChat, turnsFromChat } from "./chatReducer";
import type { ChatState, Turn } from "./chatReducer";
import type { StoredChat } from "../lib/types";

export interface ChatController {
  state: ChatState;
  streaming: boolean;
  ask: (question: string, model: ModelName, override: string | null) => Promise<void>;
  stop: () => void;
  reset: () => void;
  load: (chat: StoredChat) => void;
  retry: (model: ModelName, override: string | null) => Promise<void>;
}

export function useChat(): ChatController {
  const [state, dispatch] = useReducer(chatReducer, emptyChat);
  const chatId = useRef<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const activeTurnId = useRef<string | null>(null);
  const counter = useRef(0);
  const mounted = useRef(true);
  const streaming = state.turns.some((turn: Turn) => turn.status === "streaming");

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      controller.current?.abort();
    };
  }, []);

  const ask = useCallback(async (question: string, model: ModelName, override: string | null) => {
    counter.current += 1;
    const turnId = `turn-${counter.current}`;
    dispatch({ type: "ask", id: turnId, question });
    // A previous stream, if still running, is aborted here: only one answer
    // streams at a time. The reducer's turnId targeting means a superseded
    // stream's late events would be dropped harmlessly even without this,
    // but there is no reason to keep a connection open for an answer nobody
    // will see, so it is cancelled outright.
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    activeTurnId.current = turnId;
    try {
      const stream = streamChat(
        {
          message: question,
          model,
          ...(chatId.current === null ? {} : { chat_id: chatId.current }),
          ...(override === null ? {} : { language_override: override }),
        },
        abort.signal,
      );
      for await (const event of stream) {
        if (!mounted.current) {
          return;
        }
        if (event.type === "done") {
          chatId.current = event.data.chat_id;
        }
        dispatch({ type: "event", turnId, event });
      }
    } catch (error) {
      if (!mounted.current) {
        return;
      }
      if (abort.signal.aborted) {
        dispatch({ type: "failed", turnId, error: { code: "cancelled", message: "The answer was stopped" } });
        return;
      }
      // The message is for the log, not the interface: MessageList shows the guidance for the code.
      console.warn("chat stream failed", error);
      dispatch({
        type: "failed",
        turnId,
        error: { code: "network_error", message: "The backend did not answer" },
      });
    } finally {
      if (controller.current === abort) {
        controller.current = null;
        activeTurnId.current = null;
      }
    }
  }, []);

  const stop = useCallback(() => {
    controller.current?.abort();
    if (activeTurnId.current !== null) {
      dispatch({ type: "failed", turnId: activeTurnId.current, error: { code: "cancelled", message: "The answer was stopped" } });
    }
  }, []);

  const reset = useCallback(() => {
    controller.current?.abort();
    activeTurnId.current = null;
    chatId.current = null;
    dispatch({ type: "reset" });
  }, []);

  const load = useCallback((chat: StoredChat) => {
    controller.current?.abort();
    activeTurnId.current = null;
    chatId.current = chat.id;
    dispatch({ type: "load", chatId: chat.id, turns: turnsFromChat(chat) });
  }, []);

  const retry = useCallback(
    async (model: ModelName, override: string | null) => {
      const last = state.turns[state.turns.length - 1];
      if (last === undefined) {
        return;
      }
      await ask(last.question, model, override);
    },
    [ask, state.turns],
  );

  return { state, streaming, ask, stop, reset, load, retry };
}
