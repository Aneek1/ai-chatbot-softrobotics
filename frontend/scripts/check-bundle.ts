// Spec section 8.4: the bundle carries its own fonts and scripts, so no page of this app fetches
// from another origin. Run as: node scripts/check-bundle.ts dist
import { readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join, resolve } from "node:path";

// Outside hosts that appear as plain strings and are never fetched: XML namespace identifiers, and
// the help link React prints inside the error it throws.
export const ALLOWED_MENTIONS = ["http://www.w3.org/", "https://react.dev/errors/"];

const TEXT_SUFFIXES = new Set([".html", ".js", ".css", ".json", ".svg", ".map", ".webmanifest"]);
const MARKUP_SUFFIXES = new Set([".html", ".css", ".svg"]);
const URL_PATTERN = /(?:https?:)?\/\/[a-z0-9-]+(?:\.[a-z0-9-]+)+[^\s"'`<>)]*/gi;
const REFERENCE_PATTERN =
  /(?:\b(?:src|href|action|srcset)\s*=\s*|\burl\(\s*|\bimport\(\s*)["'`]?((?:https?:)?\/\/[^\s"'`<>)]+)/gi;
const LOCAL_HOST = /^(?:https?:)?\/\/(?:localhost|127\.0\.0\.1)(?::\d+)?(?:\/|$)/i;

export function textFilesIn(dir: string): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      found.push(...textFilesIn(path));
    } else if (TEXT_SUFFIXES.has(extname(entry))) {
      found.push(path);
    }
  }
  return found;
}

export function mentions(text: string): string[] {
  return [...text.matchAll(URL_PATTERN)]
    .map((match) => match[0])
    .filter((url) => !LOCAL_HOST.test(url));
}

export function references(text: string): string[] {
  return [...text.matchAll(REFERENCE_PATTERN)]
    .map((match) => match[1])
    .filter((url) => !LOCAL_HOST.test(url));
}

export function unexpected(urls: string[]): string[] {
  return urls.filter((url) => !ALLOWED_MENTIONS.some((allowed) => url.startsWith(allowed)));
}

function main(dir: string): void {
  const problems: string[] = [];
  const allowed = new Set<string>();
  for (const file of textFilesIn(dir)) {
    const text = readFileSync(file, "utf8");
    // Attribute and url() syntax only means one thing, so any external reference in markup fails,
    // allowlist or not. A minified bundle has no such syntax, so there it is the mention that counts.
    if (MARKUP_SUFFIXES.has(extname(file))) {
      for (const url of references(text)) {
        problems.push(`${file}: fetches ${url}`);
      }
    }
    for (const url of mentions(text)) {
      if (unexpected([url]).length > 0) {
        problems.push(`${file}: names ${url}`);
      } else {
        allowed.add(url);
      }
    }
  }
  if (problems.length > 0) {
    console.error(problems.join("\n"));
    process.exit(1);
  }
  const listed = [...allowed].sort().join(", ");
  console.log(`No external URLs in ${dir} (${allowed.size} allowed mentions: ${listed})`);
}

if (process.argv[1] !== undefined && import.meta.filename === resolve(process.argv[1])) {
  main(process.argv[2] ?? "dist");
}
