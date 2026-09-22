import { useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";

interface Props {
  onSend: (question: string) => void;
  onStop: () => void;
  streaming: boolean;
  disabled: boolean;
}

export function Composer({ onSend, onStop, streaming, disabled }: Props) {
  const [text, setText] = useState("");

  function send() {
    const question = text.trim();
    if (question === "" || disabled) {
      return;
    }
    onSend(question);
    setText("");
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    send();
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  }

  return (
    <form onSubmit={onSubmit} className="border-t border-line bg-surface p-4">
      <label htmlFor="composer" className="mb-1 block text-xs uppercase tracking-wide text-muted">
        Your question
      </label>
      <textarea
        id="composer"
        rows={3}
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Ask in any language"
        className="w-full resize-y rounded border border-line bg-paper p-3 text-ink placeholder:text-muted"
      />
      <div className="mt-2 flex items-center justify-between gap-4">
        <p className="text-xs text-muted">Enter sends. Shift and Enter start a new line.</p>
        {streaming ? (
          <button
            type="button"
            onClick={onStop}
            className="rounded border border-line px-3 py-1.5 text-sm text-ink"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={disabled || text.trim() === ""}
            className="rounded bg-accent px-3 py-1.5 text-sm text-surface disabled:opacity-50"
          >
            Send
          </button>
        )}
      </div>
    </form>
  );
}
