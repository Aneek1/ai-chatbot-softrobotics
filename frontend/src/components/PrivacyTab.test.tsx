import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { PrivacyTab } from "./PrivacyTab";
import type { EgressEntry } from "../lib/types";
import { mode } from "../test/helpers";

const entries: EgressEntry[] = [
  { time: "2026-09-16T10:00:02.000Z", host: "html.duckduckgo.com", port: 443, verdict: "blocked", component: "search" },
  { time: "2026-09-16T10:00:01.000Z", host: "127.0.0.1", port: 11434, verdict: "allowed", component: "ollama" },
];

function setup(overrides: Partial<Parameters<typeof PrivacyTab>[0]> = {}) {
  const props = {
    mode: mode(),
    entries,
    busy: false,
    onChange: vi.fn(),
    ...overrides,
  };
  render(<PrivacyTab {...props} />);
  return props;
}

describe("PrivacyTab", () => {
  it("reports the mode and whether a proxy is configured", () => {
    setup();
    expect(screen.getByRole("switch", { name: "Private mode" })).toHaveAttribute("aria-checked", "false");
    expect(screen.getByText("No proxy configured. DuckDuckGo sees the query and your address.")).toBeInTheDocument();
  });

  it("switches the mode", async () => {
    const { onChange } = setup();
    await userEvent.click(screen.getByRole("switch", { name: "Private mode" }));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("says when private mode cannot answer", () => {
    setup({ mode: mode({ private: true, ollama_reachable: false }) });
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Ollama is not reachable, so private questions will fail. There is no fallback.",
    );
  });

  it("lists the connection attempts with their verdicts", () => {
    setup();
    const rows = screen.getAllByRole("listitem");
    expect(rows[0]).toHaveTextContent("html.duckduckgo.com:443");
    expect(rows[0]).toHaveTextContent("blocked");
    expect(rows[1]).toHaveTextContent("127.0.0.1:11434");
  });

  it("says when nothing has been attempted yet", () => {
    setup({ entries: [] });
    expect(screen.getByText("No connection attempts logged yet.")).toBeInTheDocument();
  });
});
