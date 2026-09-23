# pycodecommenter-ai-backend

A small hosted proxy that lets `pycodecommenter generate --ai-draft` work
without the person running it having their own Gemini API keys. Holds the
real keys server-side (via Render's environment variable injection); the
client talks to this service instead of Gemini directly when it has no local
keys of its own.

Standalone project: does not depend on the `pycodecommenter` package. It
owns its own Gemini-calling logic (multi-key failover, per-key circuit breaker,
per-model fallback, live model discovery) in [gemini_client.py](gemini_client.py) rather than
importing it — see that module's docstring for why.

## Contract

See [docs/API.md](docs/API.md).

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Create a `.env` with your own key(s):

```
GEMINI_API_KEYS=your-key-here
```

Run it:

```bash
python app.py            # dev server, http://127.0.0.1:5000
```

Run the tests (no real network calls are made by the automated suite):

```bash
pytest .
```

## Deploying (Render)

See [render.yaml](render.yaml). Set `GEMINI_API_KEYS` as a real environment
variable in Render's dashboard — never commit it. Start command:

```
gunicorn "app:create_app()"
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEYS` | *(required)* | Comma-separated Gemini API key(s). |
| `RATE_LIMIT` | `10 per minute` | Per-IP request limit (Flask-Limiter syntax). |
| `MAX_DAILY_CALLS` | `200` | Global daily cap across all callers, in-memory (resets on redeploy). |
| `GEMINI_MODEL` | *(auto-discovered)* | Pin a specific model instead of discovering the best available ones. A pinned model has no fallback if it is overloaded. |
| `REQUEST_TIMEOUT_S` | `15` | Per-request timeout to the Gemini API. |

## Abuse/cost posture

There is no real authentication on this endpoint — anyone can read the
client's source and call it directly. The mitigations (per-IP rate limiting,
a global daily cap, zero cost exposure since Gemini's free tier is $0) bound
*nuisance*, not a security boundary. See the PyCodeCommenter project's
`Future Work/AI Backend Implementation Plan.md`, section 6.6, for the full
reasoning.
