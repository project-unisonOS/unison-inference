# Authenticated local inference packet

Status: draft for review, not operational deployment. Initial service authority
is an active issuer-signed person capability scope for local text inference.
This is narrower than the planned independent reference monitor.

Synthetic authenticated ASGI tests use a loopback RSA/JWKS and introspection
service, a signed synthetic registry and a controlled provider callback. A valid
local request invokes that callback. Missing/expired/revoked/wrong-audience or
forged identity, absent capability scope, legacy requests, remote destinations,
tools/attachments, excessive token budgets and missing trusted inventory/registry
do not invoke it. Caller cloud fallback cannot turn local outage into disclosure.

Before repair the valid synthetic baseline reproduced eight failures and one
positive local pass. The expanded full suite locally produced 51 passes and two
pre-existing Windows directory-fsync failures in model lifecycle persistence.
Canonical Linux hosted validation is required for those lifecycle cases.

This does not demonstrate real model inference, clinical correctness, provider
integration, complete principal/workload delegation or household readiness.
Remote inference remains unavailable until independently authenticated,
principal/payload/destination-bound disclosure and fallback decisions exist.
The model cannot grant that authority via a request body. Server configuration
and model execution isolation still need independent security review.

No new accounts, real credentials, grants or models have been provisioned.
Tests mint ephemeral keys only for synthetic local fixtures. Deployment requires
reviewed auth scope issuance, registry artifacts and trusted hardware inventory;
this packet does not change any operational configuration.
