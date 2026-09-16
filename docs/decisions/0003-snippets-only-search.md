# 0003: Web search reads snippets, never result pages

Status: accepted, 2026-09-15.

## Context

Web results make answers current, but fetching a result page means contacting a site the user never
chose, sending them a referrer and an IP address, and pulling in whatever that page contains. In
private mode that is exactly what the app promises not to do.

## Decision

The search interface returns titles, links and snippets from the search engine's own response.
Nothing else is fetched. Google Custom Search is used in normal mode when a key is configured,
DuckDuckGo otherwise and always in private mode.

## Consequences

- One outbound host per search, which the egress guard can allow by name.
- Answers are limited to what a snippet says, so web results support an answer rather than carry it.
- The same interface covers both engines, so the wind-down of Google's Custom Search API is a
  configuration change, not a rewrite.
