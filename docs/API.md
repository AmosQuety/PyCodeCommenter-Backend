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

## `POST /v2/draft-docstring`

Drafts several parts of one function's docstring in a single call. The
client sends the function's facts (same fields as `/v1`), the text already
known for it, and the slots it still needs:

```json
{
  "name": "total_due",
  "parameters": [
    {"name": "invoice_id", "type_hint": "str", "default": null},
    {"name": "discount", "type_hint": "float", "default": "0.0"}
  ],
  "return_type": "float",
  "is_generator": false,
  "raised_exceptions": ["KeyError", "ValueError"],
  "source": "def total_due(self, invoice_id, discount=0.0): ...",
  "known": {
    "params": {"invoice_id": "Unique identifier for the invoice."},
    "returns": null,
    "raises": {"KeyError": "If `invoice is None`."}
  },
  "slots": {
    "summary": true,
    "description": false,
    "params": ["discount"],
    "returns": true,
    "raises": ["ValueError"]
  }
}
```

`slots.params` must name real parameters and `slots.raises` must name
exceptions in `raised_exceptions`; at least one slot is required; `source`
is capped at 20,000 characters. Anything else is a **400**.

**Success — HTTP 200**, always with all five keys. Declined slots are
`null`, or absent from `params`/`raises`:

```json
{
  "summary": "Calculate the total amount due for an invoice.",
  "description": null,
  "params": {"discount": "The discount to apply to the total amount."},
  "returns": "The total amount due after tax and discount.",
  "raises": {}
}
```

Every returned value is a single line, at most 300 characters (80 for
`summary`), ending in sentence punctuation, and never contains `"""`,
`'''`, a backslash, `TODO`, or `AI-drafted`. A slot the model couldn't
answer from the code is declined rather than guessed.

## Daily allowance

Both drafting endpoints count against a per-caller daily allowance (by IP
address; 25 by default) and a shared daily cap. Every drafting response
carries:

| Header | Meaning |
|---|---|
| `X-AI-Drafts-Limit` | The caller's daily allowance. |
| `X-AI-Drafts-Remaining` | Drafts left today, after this one. |

**Allowance used up — HTTP 429**, with `Retry-After` (seconds to the next
UTC midnight):

```json
{"error": "user_daily_limit_reached", "message": "..."}
```

A malformed request (400) costs nothing. Allowances are in memory and reset
on redeploy or cold start.

## `GET /health`

Trivial `200 {"status": "ok"}`, for Render's own health monitoring and for a
client that wants to pre-warm a sleeping free-tier instance before a real
request.

## Privacy

`source` — the function's actual source text — crosses this boundary and is
forwarded to Gemini to draft from. It is never logged or persisted; only
request metadata (timestamp, latency, outcome, function name) is logged. See
the client's consent flow (`consent.py` in PyCodeCommenter) for what this means for an end
user before their code is ever sent here.
