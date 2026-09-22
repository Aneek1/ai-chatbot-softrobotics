import { describe, expect, it } from "vitest";

import { SUPPORTED_LANGUAGES, STAGE_LABELS, languageName, scriptName, scriptOf } from "./languages";

describe("language names", () => {
  it("covers the ten supported labels", () => {
    expect(Object.keys(SUPPORTED_LANGUAGES)).toHaveLength(10);
    expect(SUPPORTED_LANGUAGES.hin_Latn).toBe("Hindi (romanized)");
  });

  it("reads a label as a language name", () => {
    expect(languageName("zho_Hant")).toBe("Chinese (Traditional)");
  });

  it("leaves an unsupported label as it is", () => {
    expect(languageName("nld_Latn")).toBe("nld_Latn");
  });

  it("names the undetermined label", () => {
    expect(languageName("und")).toBe("Undetermined");
  });

  it("takes the script out of a label", () => {
    expect(scriptOf("hin_Latn")).toBe("Latn");
    expect(scriptOf("und")).toBe("");
  });

  it("names a script and falls back to its code", () => {
    expect(scriptName("Hant")).toBe("Han (Traditional)");
    expect(scriptName("Xyzz")).toBe("Xyzz");
  });

  it("names the detector stage the backend reports", () => {
    expect(STAGE_LABELS.rule).toBe("Chinese script rule");
    expect(STAGE_LABELS.specialist).toBe("Specialist detector");
  });
});
