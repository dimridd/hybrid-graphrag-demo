---
title: Authentication — JWT Issuing and Hourly Rotation
space: SEC
labels: [security, auth, jwt]
author: Hannah Brooks
updated: 2026-07-15
---
# Authentication: JWT Issuing and Hourly Rotation

## Flow
1. Users sign in with company single sign-on; accounts are provisioned from Active Directory.
2. The Auth Service issues a JWT with the user ID, tenant and group memberships.
3. orbit-gateway validates the JWT on every request, including each SSE stream.

## Rotation
Signing keys rotate every hour. The Auth Service publishes the current and previous keys on its JWKS endpoint, so tokens signed just before a rotation stay valid for their 60-minute lifetime. Rolled out in SEC-77.

## Groups and permissions
Group memberships in the JWT decide which Confluence spaces and Jira projects a user can search. The Connector Sync service stores each document's allowed groups with every chunk.

Owner: Security team (Hannah Brooks).
