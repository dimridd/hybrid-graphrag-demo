---
title: Deal Filter Pipeline — API Reference
space: ORBIT
labels: [api, sse, orbit-3]
author: Arjun Rao
updated: 2026-08-20
---
# Deal Filter Pipeline: API Reference

## Request model
Both endpoints accept a `DealFilterRequest`:

```json
{"query": "Which deals are at risk this quarter?", "tenant": "acme.prod", "deal_ids": [], "account_ids": [], "sender": "user-7f3a"}
```

## SSE streaming API — POST /sse/deals
Served by the orbit-gateway service. The response is `text/event-stream`. Events arrive in this order:

- `planning_status` — progress messages while Phase I runs (for example "Decomposing query").
- `content_delta` — one event per word-level chunk of the answer, payload `{deal_id, token}`.
- `done` — the final event, carrying the full answer, citations and `message_id`.
- `error` — sent instead of `done` when the pipeline fails; payload `{code, message}`.

The gateway sends a heartbeat comment every 15 seconds so proxies do not close idle streams.

## REST API — POST /rest/deals
Returns the same final answer as a single JSON response. Use it for batch jobs and tests, not for the chat UI.

## Error responses
- `400` — empty query or unknown tenant.
- `429` — tenant rate limit reached (30 requests per minute per tenant).
- `504` — pipeline exceeded the 90 second budget.

## Client example (Python)
```python
import httpx
with httpx.stream("POST", f"{BASE_URL}/sse/deals", json=body, timeout=None) as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            handle(line[5:])
```
