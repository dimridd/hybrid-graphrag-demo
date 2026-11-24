---
title: Enterprise Search — Connector Permission Filtering
space: SEC
labels: [security, permissions, enterprise-search]
author: Hannah Brooks
updated: 2026-10-12
---
# Enterprise Search: Connector Permission Filtering

## Principle
A user must never see text, or a fact derived from text, that they cannot open in the source system.

## Chunks
Each chunk stores the groups allowed by its source: the Confluence space permissions or the Jira project roles. At query time, the search service filters chunks by the groups in the user's JWT before ranking, not after.

## Knowledge graph facts
An entity or relationship in the knowledge graph is shown only if the user can open at least one of the chunks it was extracted from. A fact that appears only in a restricted space stays hidden even if the entity itself is public.

## Sync
The Connector Sync service re-reads permissions every 15 minutes and on every permission-change webhook. Removing a user from a space takes effect at the next sync.

## Audit
Every query logs the user, the groups used for filtering and the IDs of the returned chunks to S3 for 400 days.
