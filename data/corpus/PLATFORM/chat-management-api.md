---
title: Chat Management API — Conversations and Feedback
space: PLATFORM
labels: [api, chat, mongodb]
author: Tomás Ortega
updated: 2026-07-30
---
# Chat Management API

The Chat Service stores Orbit conversations so users can return to them. It is owned by the Platform team and stores data in MongoDB.

## How a conversation gets populated
The first question in the UI calls `POST /api/chats/create`. Every streamed answer is appended by the SSE pipeline after the `done` event (post-stream persistence).

## Data model
- `conversations` — one document per chat: `conversation_id`, `user_id`, `title`, `deal_id`, `created_at`, `last_message_at`, `message_count`, `archived`.
- `chat_messages` — one document per turn: `message_id`, `conversation_id`, `role`, `content`, `citations`, `created_at`.
- `chat_feedback` — one document per feedback event: `feedback_id`, `message_id`, `rating` (up or down), `comment`.

## Endpoints
- `POST /api/chats/list` — list conversations for a user, newest first, 20 per page with `next_cursor`.
- `POST /api/chats/create` — create a conversation.
- `POST /api/chats/rename` — rename a conversation.
- `DELETE /api/chats/{conversation_id}` — soft delete.
- `POST /api/chats/messages` — append a message.
- `POST /api/chats/feedback` — submit feedback; a second submission for the same message replaces the first.

## Known limitations
- Listing used offset pagination until PLAT-418, which caused duplicates when new chats arrived between pages. It now uses a cursor on `last_message_at`.
- There is no full-text search over messages yet.
