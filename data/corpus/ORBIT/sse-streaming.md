---
title: SSE Live-Streaming Pipeline — Architecture
space: ORBIT
labels: [sse, streaming, architecture]
author: Arjun Rao
updated: 2026-09-18
---
# SSE Live-Streaming Pipeline: Architecture and Flow

## 1. HTTP entry layer
Requests enter through the orbit-gateway service, which validates the JWT, resolves the tenant and opens an SSE response.

## 2. Tenant resolution
The tenant prefix (for example `acme` from `acme.prod`) selects a tenant config: Astra DB cluster, keyspace, enabled sources and persona. Credentials are read from the secret manager, never from the request.

## 3. Orchestrator
`run_account_deal_pipeline` is an async generator that drives Phase I (Deal Planner), Phase II (Context Builder) and Phase III (Citation Pipeline), forwarding progress as `planning_status` events.

## 4. Per-tenant runner lock
The Deal Filter Runner logs in to the CRM SDK once per tenant. Its `_shell(tenant)` method uses double-checked locking: a fast path outside the lock, then a re-check inside. This fixed the duplicate-login race tracked in ORB-2291.

## 5. Real token streaming
The Bedrock client's iterator is blocking. A background thread runs it and pushes each token onto an `asyncio.Queue` with `loop.call_soon_threadsafe`; `stream_complete` drains the queue and yields tokens as they land.

## 6. Hiding wrapper tags
The model wraps its answer in `<answer>` tags. `_stream_answer_tokens` buffers only until it knows a partial tag is not starting, so no tag text leaks to the client.

## 7. Word-level re-chunking
Provider deltas are not word aligned; one delta can hold a whole sentence. Safe text is re-split with `re.findall(r"\s*\S+", text)` and yielded word by word, adding no latency.

## 8. Final SSE contract
`planning_status` → many `content_delta` → `done` (or `error`).

## 9. Post-stream persistence
After `done`, the answer is written to the Chat Service as a fire-and-forget task, so persistence never delays the stream.
