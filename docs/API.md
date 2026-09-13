# API contract

One versioned endpoint. Gemini-agnostic on the wire — a client asks for "a
description," never "a Gemini completion" — so the backend can change or add
providers server-side without a client update.

## `POST /v1/draft-description`

```json
{
  "name": "calculate_discount",
  "parameters": [
    {"name": "price", "type_hint": "float", "default": null},
    {"name": "rate", "type_hint": "float", "default": "0.1"}
  ],
  "return_type": "float",
  "is_generator": false,
  "raised_exceptions": [],
  "source": "def calculate_discount(price, rate=0.1):\n    return price * (1 - rate)"
}
```

Only `name` is required; every other field defaults (empty parameters, `"any"`
return type, `false` generator, no exceptions, empty source) if omitted.

**Success / decline — both HTTP 200:**

```json
{"description": "Calculates the discounted price by applying rate to price."}
```

```json
{"description": null}
```

A decline is an expected outcome (low confidence, empty model response,
every key exhausted), not a server error — treat both shapes the same way on
the client: a description or nothing, never a claim to distrust.

**Malformed request — HTTP 400:**

```json
{"error": "malformed_request", "message": "Missing required field: name"}
```

**Daily cap reached — HTTP 429, with `Retry-After`:**

```json
{"error": "daily_cap_reached", "message": "..."}
```

**Per-IP rate limited — HTTP 429, with `Retry-After`:**

```json
{"error": "rate_limited", "message": "..."}
```

A client should treat any 429 as "the service has explicitly said no for
now" and fail closed rather than retry in a loop.

## `GET /health`

Trivial `200 {"status": "ok"}`, for Render's own health monitoring and for a
client that wants to pre-warm a sleeping free-tier instance before a real
request.

## Privacy

`source` — the function's actual source text — crosses this boundary and is
forwarded to Gemini to draft from. It is never logged or persisted; only
request metadata (timestamp, latency, outcome, function name) is logged. See
the client's consent flow (not yet built) for what this means for an end
user before their code is ever sent here.
