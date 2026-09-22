import type { Theme } from "../hooks/useTheme";

interface Props {
  theme: Theme;
  onToggle: () => void;
}

export function ThemeToggle({ theme, onToggle }: Props) {
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      aria-pressed={theme === "dark"}
      onClick={onToggle}
      className="rounded border border-line px-2 py-1 text-xs text-muted hover:text-ink"
    >
      Switch to the {next} theme
    </button>
  );
}
