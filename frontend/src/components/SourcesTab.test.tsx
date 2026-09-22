import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SourcesTab } from "./SourcesTab";
import { source } from "../test/helpers";

const wikipedia = source({
  id: "wikipedia:ko:1234:0",
  title: "소프트 로봇공학",
  language: "kor_Hang",
  source: "wikipedia",
  url: "https://ko.wikipedia.org/wiki/소프트_로봇공학",
  licence: "CC BY-SA 4.0",
});

describe("SourcesTab", () => {
  it("lists each document with its language and licence", () => {
    render(<SourcesTab sources={[source(), wikipedia]} web={[]} citations={null} />);
    expect(screen.getByText("Soft pneumatic actuators cast in silicone")).toBeInTheDocument();
    expect(screen.getByText("Korean")).toBeInTheDocument();
    expect(screen.getByText("CC BY-SA 4.0")).toBeInTheDocument();
  });

  it("names where the document came from", () => {
    render(<SourcesTab sources={[source(), wikipedia]} web={[]} citations={null} />);
    expect(screen.getByText("arXiv")).toBeInTheDocument();
    expect(screen.getByText("Wikipedia")).toBeInTheDocument();
  });

  it("links to the document without leaking the page it was opened from", () => {
    render(<SourcesTab sources={[source()]} web={[]} citations={null} />);
    const link = screen.getByRole("link", { name: "https://arxiv.org/abs/2401.00001" });
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noreferrer noopener");
  });

  it("marks the sources the answer cited", () => {
    render(<SourcesTab sources={[source(), wikipedia]} web={[]} citations={{ valid: [2], removed: [] }} />);
    const items = screen.getAllByRole("listitem");
    expect(within(items[1]).getByText("Cited as [2]")).toBeInTheDocument();
    expect(within(items[0]).queryByText(/Cited as/)).not.toBeInTheDocument();
  });

  it("keeps web snippets apart and says they cannot be cited", () => {
    render(
      <SourcesTab
        sources={[]}
        web={[{ title: "Casting guide", url: "https://example.invalid/g", snippet: "Degas the silicone", engine: "duckduckgo" }]}
        citations={null}
      />,
    );
    expect(screen.getByText("Web snippets (duckduckgo), never cited")).toBeInTheDocument();
    expect(screen.getByText("Casting guide")).toBeInTheDocument();
  });

  it("says plainly when an answer had no sources", () => {
    render(<SourcesTab sources={[]} web={[]} citations={null} />);
    expect(screen.getByText("No sources found for this answer.")).toBeInTheDocument();
  });
});
