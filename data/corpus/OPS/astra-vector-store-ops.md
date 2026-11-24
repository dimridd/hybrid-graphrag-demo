---
title: Astra DB Vector Store — Operations
space: OPS
labels: [astra-db, vector-store, operations]
author: Priya Shah
updated: 2026-08-22
---
# Astra DB Vector Store: Operations

## Collections
- `crm_deals` — one vector per deal, 1024 dimensions, refreshed every 30 minutes from the CRM.
- `transcripts` — call transcript chunks of about 400 tokens, written by the transcript ingestion job.
- `kb_chunks` — enterprise search chunks from Confluence and Jira, written by the Connector Sync service.

## Indexing
Embeddings are produced by the Embedding Service with the mxbai-embed-large model. Ingestion is idempotent: each chunk ID is a hash of document ID, version and chunk index, so replaying an event does not create duplicates.

## Capacity
Each tenant has its own keyspace. Alert when a keyspace exceeds 80% of its storage quota.

## Backups
Astra DB keeps daily snapshots for 14 days. Restores are run by the SRE team.
