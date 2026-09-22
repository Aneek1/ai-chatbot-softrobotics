import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { LanguageTab } from "./LanguageTab";
import { language } from "../test/helpers";

describe("LanguageTab", () => {
  it("names the detected language and keeps its raw label", () => {
    render(<LanguageTab language={language()} override={null} onOverride={vi.fn()} />);
    // The chosen language is also the top candidate, so scope the query to the detected block.
    const detected = within(screen.getByLabelText("Detected language"));
    expect(detected.getByText("Indonesian")).toBeInTheDocument();
    expect(detected.getByText("ind_Latn")).toBeInTheDocument();
  });

  it("lists the candidates the detector returned", () => {
    render(<LanguageTab language={language()} override={null} onOverride={vi.fn()} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(3);
  });

  it("shows each probability as a percentage", () => {
    render(<LanguageTab language={language()} override={null} onOverride={vi.fn()} />);
    expect(screen.getByText("62.0%")).toBeInTheDocument();
    expect(screen.getByText("31.0%")).toBeInTheDocument();
  });

  it("says which stage decided, and names the script", () => {
    render(
      <LanguageTab
        language={language({
          chosen: "zho_Hant",
          chosen_name: "Chinese (Traditional)",
          stage: "rule",
          candidates: [{ code: "zho_Hant", name: "Chinese (Traditional)", script: "Hant", probability: 0.88 }],
        })}
        override={null}
        onOverride={vi.fn()}
      />,
    );
    expect(screen.getByText("Decided by the Chinese script rule")).toBeInTheDocument();
    expect(screen.getByText("zho_Hant, Han (Traditional)")).toBeInTheDocument();
  });

  it("warns when the detection is uncertain", () => {
    render(<LanguageTab language={language({ uncertain: true })} override={null} onOverride={vi.fn()} />);
    expect(screen.getByRole("status")).toHaveTextContent(
      "The detector was not confident. The answer language falls back to the last confident message, then to English.",
    );
  });

  it("offers every supported language as an override", async () => {
    const onOverride = vi.fn();
    render(<LanguageTab language={language()} override={null} onOverride={onOverride} />);
    const select = screen.getByLabelText("Answer language");
    expect(screen.getAllByRole("option")).toHaveLength(11);
    await userEvent.selectOptions(select, "tam_Taml");
    expect(onOverride).toHaveBeenCalledWith("tam_Taml");
  });

  it("waits for a question before showing a detection", () => {
    render(<LanguageTab language={null} override={null} onOverride={vi.fn()} />);
    expect(screen.getByText("Ask something to see how its language was identified.")).toBeInTheDocument();
  });
});
