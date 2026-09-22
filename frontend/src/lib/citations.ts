const MARKER = /\s?\[(\d+)\]/g;

/** The backend drops markers that match no source; the streamed text still has them. */
export function stripRemovedCitations(text: string, removed: number[]): string {
  if (removed.length === 0) {
    return text;
  }
  const rejected = new Set(removed);
  return text.replace(MARKER, (match, digits: string) => (rejected.has(Number(digits)) ? "" : match));
}
