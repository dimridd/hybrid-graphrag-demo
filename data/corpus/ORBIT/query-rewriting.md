---
title: Follow-up Query Rewriting
space: ORBIT
labels: [rewriting, conversation, orbit-3]
author: Lena Fischer
updated: 2026-09-25
---
# Follow-up Query Rewriting in Orbit 3.0

Follow-up questions such as "what about the champion?" cannot be searched on their own. The Query Rewriter turns each follow-up into a standalone query before retrieval, using the last 4 turns of the conversation.

## How it works
1. Load the last 4 turns from the Chat Service.
2. One LLM call (Claude Haiku) returns `{standalone_query, carried_entities, is_new_topic}`.
3. If `is_new_topic` is true, earlier entities are dropped.

## Demonstration
- **Turn 1:** "Give me an overview of the Northwind Logistics deal." → sent unchanged.
- **Turn 2:** "Who is the champion?" → "Who is the champion on the Northwind Logistics deal?"
- **Turn 3:** "Has she joined a call recently?" → "Has the Northwind Logistics champion joined a call in the last 30 days?"
- **Turn 4:** "Now show me deals at risk in EMEA." → new topic; carried entities cleared.

## Known limitation
Pronouns that refer to a person mentioned 5 or more turns ago are not resolved, because only 4 turns are loaded. Tracked as ORB-2377.
