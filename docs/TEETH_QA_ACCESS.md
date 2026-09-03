# Teeth live QA access

## Purpose

The production interpreter limits anonymous users to three free interpretations. That made the permanent Teeth regression suite impossible to complete reliably. The admin-only `/admin/qa-grant` endpoint now creates an isolated, short-lived QA token without changing the public free quota, customer subscriptions, Dream Packs, rewarded-ad grants, or Stripe data.

This is QA infrastructure only. It does not change Teeth doctrine or customer access rules.

## Safety boundaries

- The endpoint requires the existing admin credential through `X-Admin-Key`.
- Grants are restricted to aliases under `@qa.jamaicantruestories.com` so real customer emails cannot be used accidentally.
- Default grant: 25 successful interpretations for 2 hours.
- Hard maximum per grant: 50 successful interpretations for 6 hours.
- Only a SHA-256 hash of the token is stored. The bearer token is returned once in the authenticated grant response.
- The protected `/qa/interpret` route invokes the normal interpreter. Failed interpretations do not deduct a use because QA access is consumed only after a successful payload is built.
- QA usage is stored in `/data/qa_grants.json`, separate from customer and anonymous entitlement stores.
- An administrator can revoke a grant immediately through `/admin/qa-revoke`.
- QA aliases must be excluded from product/revenue analytics if email-level traffic is ever aggregated into conversion reporting. These are test sessions, not sales.

## Granting a QA session

Send an authenticated POST request to `/admin/qa-grant` with JSON such as:

```json
{
  "email": "teeth-regression@qa.jamaicantruestories.com",
  "uses": 30,
  "hours": 2
}
```

Use the existing `X-Admin-Key` header. Do not place the admin key in source control, screenshots, public documentation, or customer-facing code.

The successful response returns a `grant_id`, bearer `token`, reserved QA alias, remaining uses, expiry, and hard grant limits. Keep the token out of source control, screenshots, reports, and chat.

Send the token with each `POST /qa/interpret` request using either:

```text
X-QA-Token: <token>
```

or:

```text
Authorization: Bearer <token>
```

The response labels access as `temporary_qa`, reports the QA grant balance, and confirms that customer credits were not consumed.

Revoke a grant with an authenticated POST to `/admin/qa-revoke`:

```json
{
  "grant_id": "qa-example"
}
```

`GET /qa/status` publishes non-secret route availability, hard limits, doctrine-registry verification, and exact deployment identity. `GET /version` publishes the immutable GitHub commit URL and production version without requiring customer access.

## Teeth release workflow

After this endpoint is deployed and an authenticated QA grant is active, rerun the permanent live Teeth regression suite through `/qa/interpret`. Record observed outputs and keep the readiness score at 56/100 until live evidence justifies a change. The fixed `/qa/teeth-regression` contract remains available for non-billable release smoke, but it is not a substitute for the authenticated full-path suite.
