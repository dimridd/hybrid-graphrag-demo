---
title: Incident ORB-2291 — Duplicate CRM Logins
space: OPS
labels: [incident, postmortem, deal-filter-runner]
author: Daniel Okafor
updated: 2026-08-05
---
# Incident ORB-2291: Duplicate CRM Logins and Intermittent SDK Errors

## Summary
On 2026-07-29, between 14:10 and 15:05 UTC, about 6% of Orbit deal queries failed with the SDK error `zipimport: can't read data`. Logs also showed duplicate CRM login calls for the same tenant.

## Impact
312 failed queries across 4 tenants. No data was lost.

## Root cause
Concurrent sub-queries from the runner fan-out reached the Deal Filter Runner at the same time. Each one saw no cached session for the tenant and called the CRM login, so several logins raced and one corrupted the SDK's shared cache.

## Fix
The Orbit Core team added a per-tenant lock with double-checked locking to `_shell(tenant)`: check the cache outside the lock, then re-check inside it before logging in. Deployed on 2026-07-30.

## Follow-ups
- Add an alert on duplicate logins per tenant (owner: SRE team).
- Load test with 4 concurrent sub-queries per tenant before each release.

Incident commander: Priya Shah. Component owner: Orbit Core team.
