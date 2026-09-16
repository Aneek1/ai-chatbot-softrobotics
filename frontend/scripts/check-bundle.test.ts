import { describe, expect, it } from "vitest";

import { mentions, references, unexpected } from "./check-bundle.ts";

describe("references", () => {
  it("reports a script the page would load from another origin", () => {
    const html = '<script type="module" src="https://cdn.example.com/react.js"></script>';
    expect(references(html)).toEqual(["https://cdn.example.com/react.js"]);
  });

  it("reports a stylesheet from a font host", () => {
    const html = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter">';
    expect(references(html)).toEqual(["https://fonts.googleapis.com/css2?family=Inter"]);
  });

  it("reports a font fetched from a stylesheet", () => {
    const css = "@font-face{font-family:Inter;src:url(https://fonts.gstatic.com/s/inter.woff2)}";
    expect(references(css)).toEqual(["https://fonts.gstatic.com/s/inter.woff2"]);
  });

  it("ignores relative paths and the local API", () => {
    const html = '<script src="/assets/index-abc.js"></script><img src="http://127.0.0.1:8000/api/x">';
    expect(references(html)).toEqual([]);
  });
});

describe("mentions", () => {
  it("finds a host named anywhere in a bundle, not only in an attribute", () => {
    const js = 'const endpoint="https://analytics.example.com/collect";';
    expect(unexpected(mentions(js))).toEqual(["https://analytics.example.com/collect"]);
  });

  it("allows namespace identifiers and the React error link, which are never fetched", () => {
    const js = 'e.setAttributeNS("http://www.w3.org/2000/svg","x"),t("https://react.dev/errors/418")';
    expect(mentions(js)).toHaveLength(2);
    expect(unexpected(mentions(js))).toEqual([]);
  });
});
