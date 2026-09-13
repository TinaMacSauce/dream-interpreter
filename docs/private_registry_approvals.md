# Private canonical registry approvals

Canonical text belongs in the private Sheet. New approval versions, source
captures, approval records, and real test examples must also stay private.
Public tests use generated mock annotations and provenance only.

## Runtime contract

`SNAKE_REGISTRY_APPROVALS_FILE` points to a privately provisioned JSON file outside
the repository. An unset path preserves the existing pinned snapshot contract.
A configured missing, invalid, or malformed file fails closed. Restart the
application after provisioning or replacing the file to clear cached snapshots.

The file must be approved and supplied independently by the authorized release
process. Never create approval hashes automatically from a live Sheet or accept
this configuration from requests, Sheet cells, or other untrusted content.
Neither this code nor the schema grants publication or doctrine approval.

The JSON object has exactly `schema_version` (integer `1`) and `profiles` (an
array of one to sixteen complete approved snapshots). Every profile contains:

| Field | Requirement |
| --- | --- |
| `snapshot_sha256` | `sha256:` followed by 64 lowercase hexadecimal characters; unique within the file |
| `doctrine_version` | Approved version present in that snapshot's row metadata |
| `sheet_revision` | Empty until Registrar read-back; otherwise the actual numeric source revision |
| `rule_metadata` | All existing implementation keys, with no missing or additional keys |

Each `rule_metadata` entry contains exactly `doctrine_version`, `decision_id`,
`authority`, and `updated_at_utc`. All are nonempty strings. Version and decision
must agree per row; timestamps use UTC `YYYY-MM-DDTHH:MM:SSZ`. Mixed row versions
are supported. No governing text belongs in this approval file.

Compute the digest with `registry_snapshot_sha256` over
`cluster_registry_values(values, "Snake")`: the shared header and ordered cluster
rows, string-valued cells, UTF-8 compact JSON, with Unicode preserved. Use the
same formatted string values as the application's authenticated Sheet loader.
This fingerprint covers every field, including trigger text, meaning,
precedence, safety boundaries, activation and provenance. Normalization must
not silently turn an unauthorized edit into an approved snapshot.

The validator still requires the exact schema, row width and count, known
implementation keys, matching rule IDs, fixed status/activation states, and
nonempty content and safety fields. Inactive unresolved rules cannot be enabled
through this configuration. Partial publication and arbitrary edits fail closed.
The pre-existing pinned snapshot remains accepted for rollback, provided any
configured approval file is valid.

## Consumption scope

This compatibility slice reads verified location wording from canonical rows,
retains per-row provenance, and preserves existing context keys and safety text.
It does not implement other new meanings, activation changes, or new keys.
Specific physical locations take precedence over a generic house/home match;
unmapped and unresolved locations do not acquire a house interpretation.

The public status projection excludes row text and the private approval file.
Approval files and canonical captures must not be committed, attached to public
issues or PRs, included in public CI logs, or bundled in public release artifacts.
`private/` and `*.private.json` are ignored as an additional safeguard; provision
real files outside the checkout and never force-add them.

## Controlled cutover and rollback

1. Independently verify the exact code SHA, CI result, private approved changes,
   Context oracle and QA evidence. Do not use customer credits for QA.
2. In private infrastructure, prepare the approval file from the approved source
   and complete held changeset. Reconcile its before-values with the live Sheet.
   Keep prior code and data available. Do not infer the next source revision.
3. Test the candidate against old and proposed snapshots using the authenticated
   loader with mocked transport or private staging. Public tests need no private
   credentials, approval files, or canonical captures.
4. Use the normal coordinated code/data release path. The Registrar owns all
   Sheet writes and read-back verification. The old production validator must
   not encounter a newly published snapshot during an uncontrolled cutover.
5. Record the real read-back receipt privately, reconcile the release verification
   contract, and have independent QA verify the release. Existing static release
   checks intentionally remain closed for an unreconciled new snapshot.

For code-only rollback before publication, restore the previous code and private
configuration and restart. After publication, keep the compatible consumer until
the Registrar has restored and verified the prior approved Sheet snapshot; only
then restore old code. Implementation does not write rollback data to the Sheet.

## Tests

`tests.test_snake_private_approvals` uses entirely synthetic text and metadata to
exercise approvals, tampering, mixed versions, inactive rules, parser boundaries,
rendering, source loading, cache reset and rollback. The full unit suite remains
the local regression gate. Live private validation evidence belongs in the
private worker return packet, never a public fixture or public test log.
