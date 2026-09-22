import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { MessageList } from "./MessageList";
import { newTurn } from "../state/chatReducer";
import type { Turn } from "../state/chatReducer";

function turn(changes: Partial<Turn> = {}): Turn {
  return { ...newTurn("turn-1", "Apa itu silikon?"), ...changes };
}

describe("MessageList", () => {
  it("explains what to do before anything is asked", () => {
    render(<MessageList turns={[]} selectedId={null} onSelect={vi.fn()} onRetry={vi.fn()} />);
    expect(screen.getByText("Nothing asked yet")).toBeInTheDocument();
  });

  it("shows the question and the answer so far", () => {
    render(
      <MessageList
        turns={[turn({ answer: "Silikon adalah polimer." })]}
        selectedId={null}
        onSelect={vi.fn()}
        onRetry={vi.fn()}
      />,
    );
    expect(screen.getByText("Apa itu silikon?")).toBeInTheDocument();
    expect(screen.getByText("Silikon adalah polimer.")).toBeInTheDocument();
  });

  it("announces that an answer is being written", () => {
    render(<MessageList turns={[turn()]} selectedId={null} onSelect={vi.fn()} onRetry={vi.fn()} />);
    expect(screen.getByRole("status")).toHaveTextContent("Answering");
  });

  it("shows the notices the backend sent", () => {
    render(
      <MessageList
        turns={[turn({ notices: [{ kind: "no_sources", message: "No sources found" }] })]}
        selectedId={null}
        onSelect={vi.fn()}
        onRetry={vi.fn()}
      />,
    );
    expect(screen.getByText("No sources found")).toBeInTheDocument();
  });

  it("explains an error and offers to try again", async () => {
    const onRetry = vi.fn();
    render(
      <MessageList
        turns={[turn({ status: "failed", error: { code: "ollama_unavailable", message: "Ollama is not reachable" } })]}
        selectedId={null}
        onSelect={vi.fn()}
        onRetry={onRetry}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Ollama is not reachable");
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Private mode answers with Ollama only. Start Ollama, or leave private mode to use Gemini.",
    );
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it("opens a turn in the inspector", async () => {
    const onSelect = vi.fn();
    render(
      <MessageList turns={[turn({ status: "done" })]} selectedId="turn-1" onSelect={onSelect} onRetry={vi.fn()} />,
    );
    expect(screen.getByRole("listitem")).toHaveAttribute("aria-current", "true");
    await userEvent.click(screen.getByRole("button", { name: "Inspect this answer" }));
    expect(onSelect).toHaveBeenCalledWith("turn-1");
  });

  it("names the language the answer was written in", () => {
    render(
      <MessageList
        turns={[turn({ status: "done", answerLanguage: "ind_Latn" })]}
        selectedId={null}
        onSelect={vi.fn()}
        onRetry={vi.fn()}
      />,
    );
    expect(screen.getByText("Answered in Indonesian")).toBeInTheDocument();
  });
});
