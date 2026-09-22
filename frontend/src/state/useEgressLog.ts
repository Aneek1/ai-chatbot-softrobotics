import { useEffect, useState } from "react";

import { streamEgress } from "../lib/api";
import type { EgressEntry } from "../lib/types";

/** Subscribes to GET /api/egress while `enabled`, newest entry first. */
export function useEgressLog(enabled: boolean, limit = 200): EgressEntry[] {
  const [entries, setEntries] = useState<EgressEntry[]>([]);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    const abort = new AbortController();
    void (async () => {
      try {
        for await (const entry of streamEgress(abort.signal)) {
          if (abort.signal.aborted) {
            return;
          }
          setEntries((current) => [entry, ...current].slice(0, limit));
        }
      } catch {
        // The stream ends when the backend stops or the view goes away; the list keeps what it has.
      }
    })();
    return () => abort.abort();
  }, [enabled, limit]);

  return entries;
}
