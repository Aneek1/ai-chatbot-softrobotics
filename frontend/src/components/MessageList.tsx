import { errorGuidance } from "../lib/errors";
import { languageName } from "../lib/languages";
import type { Turn } from "../state/chatReducer";

interface Props {
  turns: Turn[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRetry: () => void;
}

export function MessageList({ turns, selectedId, onSelect, onRetry }: Props) {
  if (turns.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <div className="max-w-md">
          <h2 className="text-base font-semibold text-ink">Nothing asked yet</h2>
          <p className="mt-2 text-sm text-muted">
            Ask a question in any language. The inspector shows which language was detected, which
            documents the answer came from, and what the app connected to.
          </p>
        </div>
      </div>
    );
  }

  return (
    <ol className="flex flex-col gap-8 p-6">
      {turns.map((turn) => {
        const guidance = turn.error === null ? null : errorGuidance(turn.error.code);
        return (
          <li
            key={turn.id}
            aria-current={turn.id === selectedId ? "true" : undefined}
            className={
              turn.id === selectedId
                ? "border-l-2 border-accent pl-4"
                : "border-l-2 border-transparent pl-4"
            }
          >
            <p className="whitespace-pre-wrap break-words text-sm font-medium text-ink">{turn.question}</p>
            {turn.answer !== "" && (
              <p aria-live="polite" className="mt-3 whitespace-pre-wrap break-words text-ink">
                {turn.answer}
              </p>
            )}
            {turn.status === "streaming" && (
              <p role="status" className="mt-3 text-sm text-muted">
                Answering
              </p>
            )}
            {turn.notices.map((notice, index) => (
              <p
                key={`${notice.kind}-${index}`}
                className="mt-3 rounded border border-line bg-warn-soft px-3 py-2 text-sm text-warn"
              >
                {notice.message}
              </p>
            ))}
            {turn.error !== null && (
              <div
                role="alert"
                className="mt-3 rounded border border-danger bg-danger-soft px-3 py-2 text-sm text-danger"
              >
                <p>{turn.error.message}</p>
                {guidance !== null && <p className="mt-1">{guidance}</p>}
                <button
                  type="button"
                  onClick={onRetry}
                  className="mt-2 rounded border border-danger px-2 py-1 text-xs text-danger"
                >
                  Try again
                </button>
              </div>
            )}
            <div className="mt-3 flex items-center gap-4 text-xs text-muted">
              {turn.status === "done" && turn.answerLanguage !== null && (
                <span>Answered in {languageName(turn.answerLanguage)}</span>
              )}
              <button
                type="button"
                onClick={() => onSelect(turn.id)}
                className="rounded border border-line px-2 py-1 text-xs text-muted hover:text-ink"
              >
                Inspect this answer
              </button>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
