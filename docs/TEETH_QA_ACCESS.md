# Teeth live QA access

## Purpose

The production interpreter limits anonymous users to three free interpretations. That made the permanent Teeth regression suite impossible to complete reliably. The admin-only `/admin/qa-grant` endpoint creates a short-lived, non-billable test allowance without changing the public free quota or requiring a Stripe payment.

This is QA infrastructure only. It does not change Teeth doctrine or customer access rules.

## Safety boundaries

- The endpoint requires the existing admin credential through `X-Admin-Key`.
- Grants are restricted to aliases under `@qa.jamaicantruestories.com` so real customer emails cannot be used accidentally.
- Default grant: 25 successful interpretations for 2 hours.
- Hard maximum per grant: 50 successful interpretations for 6 hours.
- The normal interpreter path is still used. Failed interpretations do not deduct a use because Dream Pack access is consumed only after a successful payload is built.
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

The successful response returns the reserved QA alias, remaining uses, expiry, and hard grant limits. The same browser/session is bound to the QA alias so subsequent `/interpret` requests exercise the normal production interpreter route.

## Teeth release workflow

After this endpoint is deployed and an authenticated QA grant is active, rerun the permanent live Teeth regression suite. Record observed outputs and keep the readiness score at 56/100 until live evidence justifies a change.
