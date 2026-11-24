---
title: Feedback Flow — Sequence
space: PLATFORM
labels: [feedback, sequence, chat]
author: Kavya Nair
updated: 2026-10-08
---
# Feedback Flow: Sequence

1. The user clicks thumbs up or down under an answer in the Orbit UI.
2. The UI calls `POST /api/v2/chats/feedback` with `sender`, `message_id`, `deal_id`, `rating` and an optional comment.
3. The Chat Service upserts one document in `chat_feedback`.
4. A nightly job exports the previous day's feedback to the evaluation bucket in S3.
5. The ML Evaluation team samples thumbs-down answers into the review queue; reviewed items can become new golden questions.

Thumbs-down rate is reported weekly. The target is below 8% of rated answers.
