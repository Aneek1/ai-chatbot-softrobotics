import { PrivateModeSwitch } from "./PrivateModeSwitch";
import { ThemeToggle } from "./ThemeToggle";
import type { Theme } from "../hooks/useTheme";
import type { Health, Mode, ModelName } from "../lib/types";

interface Props {
  health: Health | null;
  healthError: boolean;
  mode: Mode | null;
  model: ModelName;
  busy: boolean;
  theme: Theme;
  onModelChange: (model: ModelName) => void;
  onPrivateChange: (isPrivate: boolean) => void;
  onToggleTheme: () => void;
}

export function AppHeader({
  health,
  healthError,
  mode,
  model,
  busy,
  theme,
  onModelChange,
  onPrivateChange,
  onToggleTheme,
}: Props) {
  const isPrivate = mode?.private ?? false;
  const geminiAvailable = (health?.gemini_configured ?? false) && !isPrivate;
  return (
    <header
      className={
        isPrivate
          ? "flex flex-wrap items-center gap-4 border-b border-line bg-private-header px-4 py-2"
          : "flex flex-wrap items-center gap-4 border-b border-line bg-surface px-4 py-2"
      }
    >
      <h1
        className={
          isPrivate ? "text-sm font-semibold text-private-ink" : "text-sm font-semibold text-ink"
        }
      >
        Soft-robotics assistant
      </h1>
      {isPrivate && (
        <span className="flex items-center gap-1 rounded border border-accent px-2 py-0.5 text-xs text-accent">
          <svg
            viewBox="0 0 16 16"
            aria-hidden="true"
            className="h-3 w-3"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <rect x="3.25" y="7" width="9.5" height="6.5" rx="1.25" />
            <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" />
          </svg>
          Private
        </span>
      )}
      <span
        className={
          isPrivate ? "font-mono text-xs text-private-muted" : "font-mono text-xs text-muted"
        }
      >
        Detector: {health?.detector ?? (healthError ? "unreachable" : "unknown")}
      </span>
      <div className="ml-auto flex flex-wrap items-center gap-3">
        <label
          htmlFor="answer-model"
          className={
            isPrivate
              ? "text-xs uppercase tracking-wide text-private-muted"
              : "text-xs uppercase tracking-wide text-muted"
          }
        >
          Answer model
        </label>
        <select
          id="answer-model"
          value={isPrivate ? "ollama" : model}
          disabled={isPrivate}
          onChange={(event) => onModelChange(event.target.value as ModelName)}
          className="rounded border border-line bg-paper px-2 py-1 text-sm text-ink disabled:opacity-60"
        >
          <option value="ollama">Ollama</option>
          {geminiAvailable && <option value="gemini">Gemini</option>}
        </select>
        <PrivateModeSwitch isPrivate={isPrivate} busy={busy} onChange={onPrivateChange} />
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
    </header>
  );
}
