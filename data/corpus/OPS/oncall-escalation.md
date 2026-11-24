---
title: On-call and Escalation Policy
space: OPS
labels: [oncall, escalation, sre]
author: Daniel Okafor
updated: 2026-06-30
---
# On-call and Escalation Policy

## Rotations
- **SRE primary:** weekly rotation, lead Daniel Okafor. Handles alerts for orbit-gateway, Astra DB and the Chat Service.
- **Orbit Core secondary:** weekly rotation, lead Meera Kapoor. Paged for Deal Planner, Context Builder and Citation Pipeline issues.
- **Platform secondary:** lead Tomás Ortega. Paged for Chat Service and API gateway issues.

## Severity
- **SEV1:** Orbit unavailable for all tenants. Page SRE primary and both secondaries; update the status page within 15 minutes.
- **SEV2:** one tenant down or more than 5% of queries failing. Page SRE primary.
- **SEV3:** degraded quality or latency. Ticket in the owning team's Jira project.

## Postmortems
Every SEV1 and SEV2 needs a postmortem within 5 working days, written by the incident commander. Example: ORB-2291.
