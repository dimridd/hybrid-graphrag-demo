---
title: Citation Architecture — End-to-End Example
space: ORBIT
labels: [citations, architecture, grounding]
author: Meera Kapoor
updated: 2026-09-10
---
# Citation Architecture: How Orbit Answers a Question End to End

Example query: *"What's at risk on the Northwind Logistics deal, and has the champion engaged recently?"*

## Stage 1 — Label every chunk with a stable ID
Every retrieved chunk gets a short tag: T1, T2 for transcripts, E1 for emails, M1 for Echo answers, D1 for deal records. Code and the LLM agree on exactly which text each tag means; this is the anchor for every citation.

## Stage 2 — Cited mini-answers per source type
`get_evidence` turns each source type's chunks into a mini-answer where every sentence cites its tags. Transcripts with 14 chunks become 2 windows and 2 LLM calls; emails with 6 chunks fit in 1 call.

## Stage 3 — Combine into one context
Mini-answers are concatenated into a single RAG context. Raw chunks are not passed again, which keeps the final prompt under 12,000 tokens.

## Stage 4 — One final call writes the answer
The final call produces an executive summary, risks, next steps and a references block.

## Stage 5 — Verify the reference block
`verify_references` checks every cited tag against the ground-truth tag map. Unknown tags are removed and the answer is flagged. This stage was added after incident ORB-2350, where the model cited a transcript that had not been retrieved.

## Why this scales
Each stage has a bounded input size, so a question about 1 deal or 40 deals costs a predictable number of calls.
