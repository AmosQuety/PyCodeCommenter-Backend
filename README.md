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
| `RATE_LIMIT` | `10 per minute` (`5 per minute` in `render.yaml`) | Per-IP request limit (Flask-Limiter syntax). |
| `MAX_DAILY_CALLS` | `200` | Global daily cap across all callers, in-memory (resets on redeploy). |
| `MAX_DAILY_CALLS_PER_USER` | `25` | Daily drafts per caller (by IP address), in-memory (resets on redeploy). |
| `TRUSTED_PROXY_HOPS` | `1` | Proxies whose `X-Forwarded-For` entry is trusted to identify the caller: the app takes the entry that many from the right. **On Render this must be `3`** (measured: a request with no client header already arrives with three entries); the default of `1` suits a single proxy. Too high lets callers spoof their address; too low makes callers look like a proxy, and their allowance is split across whatever addresses the proxies use. The draft log line shows `forwarded_hops=N`, the number of entries seen. |
| `GEMINI_MODEL` | *(auto-discovered)* | Pin a specific model instead of discovering the best available ones. A pinned model has no fallback if it is overloaded. |
| `REQUEST_TIMEOUT_S` | `15` | Per-request timeout to the Gemini API. |

### Gemini quota and capacity

Google counts free-tier quota **per project and per model**. Measured in AI Studio on 2026-09-27 for a free project: Gemini 2.5 Flash and the Flash alias allow **5 requests a minute and 20 a day each** (Flash-Lite: 10 a minute, 20 a day). One draft is one request, so a project supplies only a few dozen drafts a day, far below the default `MAX_DAILY_CALLS`. Consequences and how the service copes:

- A 429 rests *that model on that key* for Google's `retryDelay` (an hour when the daily quota is what is spent) and the request falls through to the next model; it no longer takes the whole key out for a minute. A key is only set aside for 401/403 or an invalid key.
- `RATE_LIMIT` is set to `5 per minute` on Render so a caller paces below the per-minute limit; the client waits once on that limit and carries on.
- Set `MAX_DAILY_CALLS` and `MAX_DAILY_CALLS_PER_USER` to what your keys can really supply (models x 20 per project per day), or enable billing on the project for real capacity. Lite models are excluded from discovery on purpose (one rejected `thinkingBudget=0` with HTTP 400) and have not been re-checked.

## Abuse/cost posture

There is no real authentication on this endpoint — anyone can read the
client's source and call it directly. The mitigations (per-IP rate limiting,
a per-caller daily allowance, a global daily cap, zero cost exposure since Gemini's free tier is $0) bound
*nuisance*, not a security boundary. See the PyCodeCommenter project's
`Future Work/AI Backend Implementation Plan.md`, section 6.6, for the full
reasoning.
