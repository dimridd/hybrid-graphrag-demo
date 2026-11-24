---
title: Chat API v2 — Sender as Key
space: PLATFORM
labels: [api, chat, v2]
author: Kavya Nair
updated: 2026-10-06
---
# Chat API v2: Sender as Key (no conversation_id)

Version 2 of the Chat API removes `conversation_id` from client calls. A conversation is identified by the pair (`sender`, `deal_id`), which makes `create` idempotent.

## Endpoints
1. `POST /api/v2/chats/create` — calling twice with the same `sender` and `deal_id` returns the same conversation.
2. `POST /api/v2/chats/list` — conversation-level only; no messages are returned.
3. `GET /api/v2/chats/get` — message-level read, separate from list.
4. `POST /api/v2/chats/feedback` — keyed by `sender`, `message_id` and `deal_id`. Changing your mind overwrites the earlier rating.
5. `POST /api/v2/chats/rename` and `POST /api/v2/chats/archive` — both require `deal_id`.
6. `GET /api/v2/chats/actions` — audit log of create, rename, archive, feedback and regenerate actions, newest first.
7. `POST /api/v2/chats/regenerate` — re-runs the last question; the old answer is kept with `superseded: true`.

## Migration
v1 endpoints stay available until 2027-01-31. The Platform team tracks the migration in PLAT-455.
