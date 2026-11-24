---
title: Deal and Account Data Flow (v2)
space: ORBIT
labels: [architecture, mapreduce, orbit-3]
author: Lena Fischer
updated: 2026-09-02
---
# Deal and Account Data Flow — version 2

This page describes how `deal_ids`, `account_ids`, transcripts, emails and Echo meeting-assistant answers are processed in Orbit 3.0.

## 1. Entry
The caller supplies a query plus optional `deal_ids` and `account_ids`. If none are supplied, the Deal Planner finds them.

## 2. Phase I — Planning
The Deal Planner classifies intent and decomposes the query (see the Deal Filter Pipeline page).

## 3. Phase II — Context Assembly
`build_contexts` is an async generator owned by the Context Builder component. It assembles four context types:
- Deal contexts from CRM fields.
- Transcript contexts from Astra DB, ordered by call time.
- Email contexts from the email index.
- Echo responses: answers previously given by the Echo meeting assistant.
Empty keys are pruned before Phase III.

## 4. Phase III — Concurrent MapReduce
Each context type is handled by its own MapReduce Engine.
- **Fast path:** if the context fits in `max_context_chars` (48,000), one LLM call produces the evidence.
- **Overflow path:** the context is split into windows; each window is mapped to a partial answer, and a reduce call merges them. A map worker that finds nothing returns the token `NOTHING`, which the reducer ignores.
Back-pressure: at most 8 map calls run at once across all engines, controlled by an asyncio semaphore.

## 5. Phase IV — Synthesis
Answers that start with "No ", "Error" or "No relevant" are excluded before synthesis. The Citation Pipeline then writes the final answer.

## 6. Key limits
| Limit | Value |
|---|---|
| Max transcript chars per deal | 80,000 |
| max_context_chars per map call | 48,000 |
| Concurrent map calls | 8 |
