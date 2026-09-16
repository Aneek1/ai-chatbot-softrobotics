# 0004: Private mode is app-wide, and never falls back

Status: accepted, 2026-09-15.

## Context

A per-chat privacy setting is easy to get wrong: a background request, a health check or a retry can
leave through a path the user thought was closed, and the user cannot see which state a given
request ran in.

## Decision

One app-wide switch. Turning it on waits for answers in flight, then applies. In private mode the
answer model is Ollama, search is DuckDuckGo or nothing, no history is written, and the egress guard
allows only loopback, the Ollama host and the search host or the configured proxy. A request that
cannot be served privately fails with an error; it never falls back to a non-private path.

## Consequences

- The privacy state is a single, visible thing: the header shows it, `GET /api/mode` reports it, and
  `GET /api/egress` streams every connection attempt with its verdict.
- With Ollama unreachable, private mode answers nothing. That is the intended behaviour.
- Switching modes waits for running answers, so a switch can take as long as an answer takes.
- The guard is in-process. It does not cover the browser or anything else on the machine, which the
  README states next to the promise.
