---
title: Orbit 3.0 — Deal Filter Pipeline
space: ORBIT
labels: [architecture, pipeline, orbit-3]
author: Meera Kapoor
updated: 2026-08-14
---
# Orbit 3.0: Deal Filter Pipeline

## Overview
The Deal Filter Pipeline is a two-phase AI system that answers natural-language questions about sales deals, such as *"Which deals are at risk this quarter?"*. It combines CRM metadata, call transcripts and an LLM served through AWS Bedrock to produce one grounded markdown brief. The pipeline is owned by the Orbit Core team.

## Phase I — Plan & Retrieve
The entry point is `DealPlanner.run(query)`.

### Step I-1 — Intent Classification
`_classify_intent` makes one LLM call (Claude Haiku on Bedrock) and returns a JSON object with `is_deal_related`, `intent` and `confidence`. Queries that are not deal related exit early with a clarifying message.

### Step I-2 — Query Decomposition
`_decompose_query` makes one LLM call that splits the question into at most 6 independent sub-queries. Rules enforced in the prompt: every sub-query must be answerable from CRM fields alone, and no sub-query may repeat a filter of another.

### Step I-3 — Runner Fan-out
`_run_sub_queries` sends each sub-query to the Deal Filter Runner, which performs a vector similarity search against the Astra DB CRM collection. This step makes no LLM calls. Concurrency is capped by `_RUNNER_CONCURRENCY = 4`.

### Steps I-4 and I-5 — Aggregation
Results are merged per `deal_id` and serialised into `planner_output`.

## Bridge — Memory Extraction
`extract_deal_memory(planner_output)` collects the deal IDs and their CRM fields. `_fetch_ordered_chunks` then loads transcript chunks for those deals from Astra DB in call order, capped at 80,000 characters per deal.

## Phase II — Synthesis
1. Build transcripts per deal.
2. Invert results into `sub_query_groups`.
3. For each group, concurrently: reformat the sub-query (1 LLM call) and synthesise a group answer (1 LLM call).
4. Accumulate the final answer (1 LLM call) using Claude Sonnet on Bedrock.

The output is `{original_query, final}`. A typical query makes 2 + 2·G + 1 LLM calls, where G is the number of groups.
