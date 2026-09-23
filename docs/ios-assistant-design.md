# An on-device assistant for iOS

A design, not a plan. It records what the thing is, where the network boundary sits, what has to be
declared to Apple, and in what order to build it. The Core ML conversion it depends on is in
[coreml-lid-task.md](coreml-lid-task.md).

## What it is

An assistant that runs on the phone. It identifies the language you write in, retrieves from a
knowledge base shipped inside the app, and answers with a model on the device. Web search is **on by
default** and adds current information the shipped corpus cannot have.

## The network boundary

There is no such thing as a one-way connection. Fetching anything means sending a request, and that
request carries the query, an IP address, a TLS SNI and a timing. So this design does not claim that
nothing leaves. It claims something checkable instead:

> Answering from the shipped corpus requires no network at all. Web search sends a query and
> nothing else. Everything sent is written to a log the user can read. Private mode stops it, and
> never silently falls back.

That is the same posture as the desktop application, and the reasoning is already recorded in
[decisions/0003-snippets-only-search.md](decisions/0003-snippets-only-search.md) and
[decisions/0004-app-wide-private-mode.md](decisions/0004-app-wide-private-mode.md).

**What stays on the device, always:** the question, the conversation, the retrieved passages, the
answer, the index, the history, the language-identification result.

**What leaves, when search is on:** one search query per question, to one host. Never the
conversation, never the retrieved passages, never the answer.

**What is never fetched:** result pages. Only the snippets the search engine itself returns, per
decision 0003 — fetching a result page means contacting a site the user never chose.

### Decide this before writing any code

**What exactly goes in the outbound query.** The desktop application sends the user's question. On a
general assistant that is a heavier decision than on a soft-robotics corpus: a question in Hindi
about a medical symptom is not the same class of data as three keywords about actuators. The options
are the raw question, a keyword reduction, or a model-rewritten query — and a rewrite performed on
the device leaks less but costs a generation pass. Choose deliberately and write the choice down,
because it is hard to change once users rely on the results.

## What Apple has to be told

Search on by default means the app transmits user-entered text to a third party by default. That is
a declaration, not a courtesy:

- **Privacy nutrition labels** at submission must say that user content is collected or transmitted
  and for what purpose.
- **A privacy manifest** (`PrivacyInfo.xcprivacy`) must declare the data types and any required-reason
  APIs used.
- If a third-party search SDK is ever added it needs its own manifest and signature. Calling a search
  endpoint over plain HTTPS avoids that entirely, which is a reason to prefer it.

The connection log is, usefully, already an exact description of what to declare. Most applications
have to estimate this; here it can be read off.

## Build it in stages

Each stage ships something on its own. Do not start the next until the previous one runs on a
device.

### Stage 1 — Language identification, offline (a weekend)

A text field. Type in any of the ten supported languages and watch the identification update live,
with candidate probabilities, entirely offline.

- Core ML conversion of `lid-specialist-e5-head.onnx` (**0.8 MB**) and multilingual-e5-small
- No network, no model server, no corpus
- Proves the Core ML path end to end and is a demonstrable artifact by itself

This is the stage that matters most for an on-device role, and the smallest.

### Stage 2 — Retrieval, still offline

Ship the index inside the app and answer with passages, no generation.

- The corpus is tiny: 136 KB of JSONL, a 670 KB index
- Embed the query on the device with the same e5 model Stage 1 already loads
- Cosine similarity over a few hundred vectors needs no vector database — a plain array will do
- Still **zero network**

### Stage 3 — Generation on the device

A small quantized model via MLX or llama.cpp with Metal.

**The limit is memory, not compute.** iOS gives each app a jetsam budget; exceeding it kills the app
rather than slowing it. Even on a 12–16 GB device an 8B model is not a comfortable fit, and the
Increased Memory Limit entitlement moves that line without removing it. Plan for **1–3B quantized**,
and measure resident memory on a real device early — the simulator will mislead you.

### Stage 4 — Web search, on by default

- One Swift module is the only place in the application that opens a network connection, mirroring
  `frontend/src/lib/api.ts` in the web client. That is what makes the boundary testable.
- Every attempt is appended to a connection log the user can open.
- Private mode turns it off and is enforced, not requested.
- First run states plainly that search is on and what it sends.

## Swift, for someone who has written C

You have written a 2,500-line GTK3 desktop shell in C. Swift will feel less like that than you
expect, and the parts that transfer are not the ones you would guess.

**You do not need Objective-C.** Modern iOS is Swift and SwiftUI. Objective-C appears only when
touching old frameworks, and nothing here requires it.

**What is different from C, and simpler:**

- **No manual memory management.** Automatic reference counting handles it. No `malloc`, no `free`,
  no ownership rules to carry in your head.
- **Optionals.** A value that may be absent has a distinct type, and the compiler forces you to deal
  with it. This is the one genuinely new idea, and it removes the entire class of null-pointer bugs
  you are used to guarding against by hand.
- **Structs are values, not pointers.** Assigning copies. No aliasing surprises.
- **Bounds-checked collections**, string handling that understands Unicode, and no header files.

**What transfers from the React work you just finished, which is more than you think:** SwiftUI is
declarative and state-driven. You describe what the view should look like for a given state and the
framework re-renders when the state changes. `@State` is `useState`. That mental model is the hard
part of SwiftUI, and you already have it.

**What transfers from C:** feeding a model. Core ML takes an `MLMultiArray`, which is a typed buffer
with a shape — laying out 384 floats correctly is exactly the kind of thing you have already done.

**How much Swift Stage 1 actually needs:** a text field, a debounce, a call into Core ML, and a list
of results. On the order of 200 lines. You are not learning Swift as a language project; you are
learning enough of it to call a model and show the answer.

## What this is not

- **Not the desktop application ported.** The FastAPI backend, Ollama and embedded Qdrant do not run
  on iOS, and no chip changes that — it is a matter of what the platform permits, not how fast it is.
- **Not a general-knowledge assistant, yet.** The corpus is small, licensed and documented, which is
  what makes a cited answer meaningful. Widening it is a data-provenance project in its own right,
  not a switch to flip.
- **Not claiming nothing leaves the device.** See the boundary above.
