# PDF Pal

A fully local PDF Q&A app for organizations that can't send documents to cloud AI services. Upload a PDF, ask questions, get cited answers - nothing leaves your local machine.

Capable of running entirely on CPU - GPU not required.

At this time, its not suited for multi-user workloads, but is capable of running in a server environment behind Caddy or your flavor of reverse proxy. There is NO auth support at this time, consider something like Cloudflare Access if required.

---

## Requirements

- [Docker](https://www.docker.com/) with Compose
- ~5 GB disk space for models on first run
- An NVIDIA GPU is supported but not required — CPU mode works out of the box

---

## Quick start

```bash
git clone https://github.com/your-username/pdf-pal.git
cd pdf-pal
docker compose up --build
```

Open **http://localhost:7842** in your browser.
> this port is configurable via the .env file

Models are pulled from Ollama automatically on first launch. This may take a few minutes. Subsequent starts are instant — models are cached in a Docker volume.

---

## Configuration

All settings live in `.env` at the project root:

```env
APP_PORT=7842               # Host port the app is exposed on

CHAT_MODEL_GPU=llama3.1:8b  # Model used when USE_GPU=true
CHAT_MODEL_CPU=llama3.2:3b  # Model used when USE_GPU=false
EMBED_MODEL=nomic-embed-text

USE_GPU=false               # Set to true if you have an NVIDIA GPU
```

**Recommended chat models:**

| Mode | Options |
|------|---------|
| CPU  | `llama3.2:3b`, `llama3.2:1b`, `phi4-mini` |
| GPU  | `llama3.1:8b`, `gemma3:12b`, `mistral:7b` |

> Avoid reasoning models — they are significantly slower and not suited for Q&A workloads.

Changes to `.env` take effect on the next `docker compose up`.

To remove all data including cached models:

```bash
docker compose down -v
```

---

## Privacy & safety

- **100% local** — no data is sent to any external service. All LLM inference runs on your own hardware via [Ollama](https://ollama.com).
- **Session isolation** — each session gets its own database. Starting a new session wipes all uploaded documents and embeddings.
- **No accounts or tracking** — no logins, no telemetry, no persistent user data.
- **50 MB upload limit** — enforced server-side.
- **Read-only PDF parsing** — documents are parsed for text only; original files are not stored.
