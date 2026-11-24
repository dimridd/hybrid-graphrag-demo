---
title: Follow-up Query Evaluation — Context Retention
space: EVAL
labels: [evaluation, follow-up, conversation]
author: Rohan Mehta
updated: 2026-10-02
---
# Follow-up Query Evaluation: Context Retention

We tested whether Orbit keeps deal context across a 3-turn conversation about the Northwind Logistics deal.

## Turn 1 — Deal overview
The answer covered the deal summary, key stakeholders (champion: Grace Lindqvist, VP Operations), value drivers and next steps. Assessment: strong foundation for follow-ups.

## Turn 2 — "Tell me the major pain points"
The Query Rewriter carried the deal entity forward. The answer listed 4 deal-specific pain points instead of generic ones. Assessment: strong.

## Turn 3 — "Which is the biggest one? Draft an email about it"
The answer picked manual route planning as the biggest pain point and drafted an email to Grace Lindqvist. Issue found: the biggest pain point did not match the order used in turn 2. Root cause: turn 2's answer was not passed back as context, only the rewritten query. Fix tracked as ORB-2381.

## Summary
Context retention: 3 of 3 turns kept the right deal. Consistency across turns: 2 of 3.
