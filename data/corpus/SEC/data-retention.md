---
title: Data Retention and PII Handling
space: SEC
labels: [security, pii, retention]
author: Hannah Brooks
updated: 2026-05-20
---
# Data Retention and PII Handling

- Chat messages are kept for 365 days, then deleted by a nightly job in the Chat Service.
- Call transcripts follow the customer's CRM retention setting, at most 2 years.
- Query audit logs are kept for 400 days in S3.
- Prompts sent to Bedrock are not stored by the provider; Bedrock model invocation logging is disabled.
- Email addresses and phone numbers are masked in transcript chunks before embedding.
- Deleting a document in Confluence removes its chunks and every graph fact whose only source was that document at the next sync.
