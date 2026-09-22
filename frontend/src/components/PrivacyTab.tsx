import { PrivateModeSwitch } from "./PrivateModeSwitch";
import type { EgressEntry, Mode } from "../lib/types";

interface Props {
  mode: Mode | null;
  entries: EgressEntry[];
  entriesFailed: boolean;
  busy: boolean;
  onChange: (isPrivate: boolean) => void;
}

export function PrivacyTab({ mode, entries, entriesFailed, busy, onChange }: Props) {
  if (mode === null) {
    return <p className="text-sm text-muted">Waiting for the backend.</p>;
  }
  return (
    <div className="flex flex-col gap-5">
      <PrivateModeSwitch isPrivate={mode.private} busy={busy} onChange={onChange} />
      <p className="text-sm text-muted">
        {mode.proxy_configured
          ? "Private searches go through the configured proxy, which resolves the host name."
          : "No proxy configured. DuckDuckGo sees the query and your address."}
      </p>
      {mode.private && !mode.ollama_reachable && (
        <p role="alert" className="rounded border border-danger bg-danger-soft px-3 py-2 text-sm text-danger">
          Ollama is not reachable, so private questions will fail. There is no fallback.
        </p>
      )}
      <div>
        <h3 className="text-xs uppercase tracking-wide text-muted">Connection attempts</h3>
        {entriesFailed && (
          <p role="alert" className="mt-2 text-sm text-danger">
            The connection log could not be loaded, so this list may be incomplete.
          </p>
        )}
        {entries.length === 0 ? (
          !entriesFailed && <p className="mt-2 text-sm text-muted">No connection attempts logged yet.</p>
        ) : (
          <ul className="mt-2 flex flex-col gap-1">
            {entries.map((entry) => (
              <li
                key={`${entry.time}-${entry.host}-${entry.port}`}
                className="flex items-baseline justify-between gap-3 font-mono text-xs"
              >
                <span className="truncate text-ink">
                  {entry.host}:{entry.port ?? "-"}
                </span>
                <span className={entry.verdict === "blocked" ? "text-danger" : "text-muted"}>
                  {entry.verdict}
                </span>
                <span className="text-muted">{entry.component}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
