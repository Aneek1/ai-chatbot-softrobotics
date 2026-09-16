# 0002: Two-stage detection with a trained specialist

Status: accepted, 2026-09-15.

## Context

A single general detector is weak exactly where this app is judged: Malay against Indonesian,
Simplified against Traditional Chinese, and romanized Hindi against Urdu and English. Those pairs
share vocabulary or a script, and short questions give a detector little to work with.

## Decision

Run the general detector first. When its top label falls inside one of the three confusion groups,
ask a specialist trained in this repository to choose within that group, and ignore any label it
returns from outside the group. Until a specialist file is configured, Han-script text falls back to
an OpenCC conversion rule, reported as stage `rule` so the interface can say which stage decided.

## Consequences

- The confusion groups can be improved without touching the general detector or its coverage.
- Every detection carries the stage that produced it, so a wrong answer can be traced.
- There is one more model file to download, and a specialist adds latency to questions that land in
  a group. Both are measured in `results/langid-eval-2026-09-15.json`.
- The specialist's training and selection are described in section 16 of the design document; it is
  chosen on validation data and evaluated once on held-out sets.
