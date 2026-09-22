import { describe, expect, it } from "vitest";

import { stripRemovedCitations } from "./citations";

describe("stripRemovedCitations", () => {
  it("removes a marker the backend rejected, with the space before it", () => {
    expect(stripRemovedCitations("Silicone is cast [1]. It cures [9].", [9])).toBe(
      "Silicone is cast [1]. It cures.",
    );
  });

  it("keeps markers that match a source", () => {
    expect(stripRemovedCitations("Cast in a mould [2].", [])).toBe("Cast in a mould [2].");
  });

  it("removes every occurrence of a rejected number", () => {
    expect(stripRemovedCitations("A [4]. B [4].", [4])).toBe("A. B.");
  });
});
