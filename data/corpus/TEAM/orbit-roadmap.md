---
title: Orbit Roadmap — Q4 2026
space: TEAM
labels: [roadmap, planning, orbit-3]
author: Sofia Alvarez
updated: 2026-10-05
---
# Orbit Roadmap: Q4 2026

1. **Skip decomposition for simple questions** — use intent confidence above 0.9 to skip `_decompose_query`. Expected to fix single-deal regressions from the Orbit 2 vs Orbit 3 report. Owner: Orbit Core team.
2. **Pass previous answers to the Query Rewriter** — fixes the turn-consistency issue (ORB-2381). Owner: Lena Fischer.
3. **Chat API v2 migration** — move all clients off v1 before 2027-01-31 (PLAT-455). Owner: Platform team.
4. **Graph search in enterprise search** — add a knowledge graph next to BM25 and vector search for multi-hop questions such as "who owns the component in this incident?". Owner: Platform team with the ML Evaluation team.
5. **Duplicate-login alert** — follow-up from ORB-2291. Owner: SRE team.
