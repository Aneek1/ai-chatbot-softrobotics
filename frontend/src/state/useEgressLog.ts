import { useEffect, useState } from "react";

import { streamEgress } from "../lib/api";
import type { EgressEntry } from "../lib/types";

export interface EgressLog {
  entries: EgressEntry[];
  /** True when the stream ended in an error, not a clean close (e.g. unmount/abort). */
  failed: boolean;
}

/** Subscribes to GET /api/egress while `enabled`, newest entry first. */
export function useEgressLog(enabled: boolean, limit = 200): EgressLog {
  const [entries, setEntries] = useState<EgressEntry[]>([]);
  const [failed, setFailed] = useState(false);

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
        // An abort (view went away) is an intentional, clean stop, not a failure to report.
        if (!abort.signal.aborted) {
          setFailed(true);
        }
      }
    })();
    return () => abort.abort();
  }, [enabled, limit]);

  return { entries, failed };
}
