# Jarvis

A personal AI workspace: chat, RAG over your documents, spreadsheet analysis, web search, memory, and voice — one agent, running locally.

The application lives in [`docchat/`](docchat/). See [`docchat/README.md`](docchat/README.md) for setup, features, and architecture.

> **Security:** API keys are read from environment variables or `docchat/.env` / the Settings UI — never committed. `docchat/data/` (database, uploads, saved keys) is gitignored.
