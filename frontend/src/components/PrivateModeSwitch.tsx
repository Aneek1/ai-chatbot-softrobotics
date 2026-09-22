interface Props {
  isPrivate: boolean;
  busy: boolean;
  onChange: (isPrivate: boolean) => void;
}

export function PrivateModeSwitch({ isPrivate, busy, onChange }: Props) {
  return (
    <button
      type="button"
      role="switch"
      aria-label="Private mode"
      aria-checked={isPrivate}
      disabled={busy}
      onClick={() => onChange(!isPrivate)}
      className="flex items-center gap-2 rounded border border-line px-2 py-1 text-xs text-ink disabled:opacity-50"
    >
      <span
        aria-hidden="true"
        className={isPrivate ? "h-2 w-2 rounded-full bg-accent" : "h-2 w-2 rounded-full bg-line"}
      />
      Private mode {isPrivate ? "on" : "off"}
    </button>
  );
}
