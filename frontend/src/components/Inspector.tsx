import { useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";

export interface InspectorPanel {
  id: string;
  label: string;
  content: ReactNode;
}

interface Props {
  panels: InspectorPanel[];
  onClose?: () => void;
}

export function Inspector({ panels, onClose }: Props) {
  const [activeId, setActiveId] = useState(panels[0].id);
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const active = panels.find((panel) => panel.id === activeId) ?? panels[0];

  function move(offset: number) {
    const index = panels.findIndex((panel) => panel.id === active.id);
    const next = panels[(index + offset + panels.length) % panels.length];
    setActiveId(next.id);
    buttons.current.get(next.id)?.focus();
  }

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      move(1);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      move(-1);
    }
  }

  return (
    <section aria-label="Inspector" className="flex h-full flex-col bg-surface">
      <div className="flex items-center justify-between border-b border-line px-2">
        <div role="tablist" aria-label="Inspector tabs" className="flex">
          {panels.map((panel) => (
            <button
              key={panel.id}
              ref={(element) => {
                if (element === null) {
                  buttons.current.delete(panel.id);
                } else {
                  buttons.current.set(panel.id, element);
                }
              }}
              type="button"
              role="tab"
              id={`tab-${panel.id}`}
              aria-controls={`panel-${panel.id}`}
              aria-selected={panel.id === active.id}
              tabIndex={panel.id === active.id ? 0 : -1}
              onClick={() => setActiveId(panel.id)}
              onKeyDown={onKeyDown}
              className={
                panel.id === active.id
                  ? "border-b-2 border-accent px-3 py-2 text-sm text-ink"
                  : "border-b-2 border-transparent px-3 py-2 text-sm text-muted hover:text-ink"
              }
            >
              {panel.label}
            </button>
          ))}
        </div>
        {onClose !== undefined && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close the inspector"
            className="rounded px-2 py-1 text-xs text-muted hover:text-ink"
          >
            Close
          </button>
        )}
      </div>
      <div
        role="tabpanel"
        id={`panel-${active.id}`}
        aria-labelledby={`tab-${active.id}`}
        tabIndex={0}
        className="flex-1 overflow-y-auto p-4"
      >
        {active.content}
      </div>
    </section>
  );
}
