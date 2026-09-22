import type { ChatSummary } from "../lib/types";

interface Props {
  chats: ChatSummary[];
  currentId: string | null;
  isPrivate: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export function ChatList({ chats, currentId, isPrivate, onNew, onSelect, onDelete }: Props) {
  return (
    <nav aria-label="Chats" className="flex h-full flex-col border-r border-line bg-surface">
      <div className="flex items-center justify-between border-b border-line px-3 py-2">
        <h2 className="text-xs uppercase tracking-wide text-muted">Chats</h2>
        <button
          type="button"
          onClick={onNew}
          className="rounded border border-line px-2 py-1 text-xs text-ink"
        >
          New chat
        </button>
      </div>
      {chats.length === 0 ? (
        <p className="px-3 py-4 text-sm text-muted">
          {isPrivate
            ? "Private chats stay in memory and are cleared when private mode ends."
            : "No chats yet. Ask a question to start one."}
        </p>
      ) : (
        <ul className="flex-1 overflow-y-auto">
          {chats.map((chat) => (
            <li key={chat.id} className="flex items-center gap-1 border-b border-line px-2 py-1">
              <button
                type="button"
                aria-label={`Open chat: ${chat.title}`}
                aria-current={chat.id === currentId ? "true" : undefined}
                onClick={() => onSelect(chat.id)}
                className={
                  chat.id === currentId
                    ? "flex-1 truncate rounded bg-accent-soft px-2 py-1 text-left text-sm text-ink"
                    : "flex-1 truncate rounded px-2 py-1 text-left text-sm text-muted hover:text-ink"
                }
              >
                {chat.title}
              </button>
              <button
                type="button"
                aria-label={`Delete chat: ${chat.title}`}
                onClick={() => onDelete(chat.id)}
                className="rounded px-2 py-1 text-xs text-muted hover:text-danger"
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </nav>
  );
}
