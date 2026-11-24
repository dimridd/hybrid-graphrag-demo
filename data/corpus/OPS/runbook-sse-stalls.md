---
title: Runbook — SSE Streams Stall or Arrive All at Once
space: OPS
labels: [runbook, sse, streaming]
author: Priya Shah
updated: 2026-09-12
---
# Runbook: SSE Streams Stall or Arrive All at Once

## Symptoms
- The Orbit UI shows "Planning…" for a long time, then the whole answer appears at once.
- Or the stream stops after 60 seconds with no `done` event.

## Checks
1. Call `POST /sse/deals` directly with curl and `-N`. If tokens arrive smoothly, the problem is in front of the gateway.
2. Check the load balancer: response buffering must be off for `/sse/*` routes. Buffering was the cause of ORB-2310.
3. Check that heartbeats arrive every 15 seconds. If not, the orbit-gateway pod may be CPU throttled.
4. Check Bedrock latency on the dashboard. Time to first token above 8 s usually means Bedrock throttling.

## Fixes
- Buffering: set `X-Accel-Buffering: no` and disable proxy buffering for the route.
- Idle timeout: raise the load balancer idle timeout to 120 seconds.
- Throttling: switch the tenant to the secondary Bedrock region.

Escalate to the Platform team if the gateway itself is slow, or to the Orbit Core team if Phase I takes more than 20 seconds.
