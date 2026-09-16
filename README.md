# ai-chatbot-softrobotics

This repository is being rebuilt from a Tkinter prototype into a multilingual retrieval assistant for soft-robotics fabrication, with language identification at its core.

The design is in [docs/design/2026-09-15-multilingual-rag.md](docs/design/2026-09-15-multilingual-rag.md). Setup instructions will be added once the backend runs end to end.

## Private mode

Private mode is an app-wide switch (`PUT /api/mode`, or `PRIVATE_MODE=true` at startup).
While it is on:

- Answers come from Ollama on your own machine. Gemini is never called, and a request that
  cannot be answered privately fails instead of falling back.
- Web search goes to DuckDuckGo, and only the search page itself is fetched: the assistant
  reads titles, links and snippets, and never opens a result page.
- Chats are kept in memory. Nothing is written to `data/history.db`, and the private chats are
  cleared when you switch back or restart.
- Question text is kept out of the server log.

An egress guard runs inside the backend process. It checks every DNS lookup and every socket
connection against a list of allowed hosts: loopback and your Ollama host at all times,
`generativelanguage.googleapis.com` and `www.googleapis.com` in normal mode, and
`html.duckduckgo.com` (or your proxy) in private mode. A blocked name is refused before any DNS
query is sent, and every attempt is logged. `GET /api/egress` streams that log, so you can watch
what the app connects to while you use it.

Set `PRIVATE_PROXY=socks5h://host:port` to send private searches through a SOCKS5 proxy, such as
Tor. With `socks5h`, the proxy resolves the host name, so the search host is never looked up from
your machine, and the proxy becomes the only outside host the guard allows.

### What private mode does not do

- The guard covers this application only. It is not a firewall: your browser, your other
  programs, and anything else on the machine are unaffected.
- Without a proxy, DuckDuckGo still sees your query and your IP address.
- The Ollama host you configure in `OLLAMA_URL` is allowed in both modes. If you point it at
  another machine, your private questions go to that machine.
- The guard works by wrapping Python's socket functions, so it only sees libraries that use
  them. The app's HTTP clients do; a library with its own network stack in C or Rust would not
  be covered, which is why the search client here is plain `httpx`.
- Stronger isolation needs to come from outside the app, by running it on a network that only
  reaches Ollama. The container setup for that is documented with the Docker image.
