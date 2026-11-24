---
title: Orbit 2 vs Orbit 3 — Evaluation Report
space: EVAL
labels: [evaluation, report, orbit-3]
author: Rohan Mehta
updated: 2026-09-30
---
# Orbit 2 vs Orbit 3: Evaluation Report

## Result
Orbit 3 wins the majority of scored questions: it is ahead on 34 of 58 questions (58.6%) for `context_recall` and 33 of 58 for `answer_faithfulness`. Orbit 2 is ahead on 15; the rest are ties.

## Method
- Golden set: 58 questions written by sales engineers, stratified into fact, multi-deal and follow-up questions (see Golden Set Methodology).
- Both versions answered every question with the same CRM snapshot.
- Metrics: `context_recall` (share of expert facts present in retrieved context), `answer_faithfulness` (LLM judge, GPT-4o) and latency to first token.

## Latency
Median time to first token fell from 6.8 s in Orbit 2 to 2.9 s in Orbit 3, because Orbit 3 streams with SSE instead of waiting for the full answer.

## Where Orbit 2 still wins
Single-deal fact questions, where Orbit 3's query decomposition adds a call without adding context. Proposed fix: skip decomposition when intent confidence is above 0.9.
