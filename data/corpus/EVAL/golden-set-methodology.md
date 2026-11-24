---
title: Golden Set Methodology
space: EVAL
labels: [evaluation, golden-set, methodology]
author: Rohan Mehta
updated: 2026-08-28
---
# Golden Set Methodology

## Purpose
The golden set is the fixed list of questions used to compare versions of Orbit and the enterprise search index before release.

## How questions are written
- Sales engineers and the ML Evaluation team write questions they really ask.
- Each question stores a reference answer and evidence quotes copied from the source documents. A retrieved chunk counts as relevant if it contains an evidence quote.
- Questions are stratified by type: fact, process, multi-hop and aggregation, so a release cannot look good by improving only easy questions.

## Metrics
- Recall@10: share of evidence found in the top 10 results.
- nDCG@10: rewards finding evidence near the top.
- MRR: position of the first relevant result.
- LLM judge (GPT-4o) for faithfulness and correctness of the final answer.

## Release gate
A release ships only if Recall@10 and nDCG@10 do not drop by more than 0.02 against the current version. Thumbs-down answers from the feedback flow are reviewed monthly and added as new questions.
