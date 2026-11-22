# Per-Tenant Fair-Share Scheduling

`--tenant-config` adds a per-tenant fair-share layer on top of the existing
FCFS and priority queues. It is for deployments that serve several tenants from
one engine and need each tenant to get its own share of the cluster, while
still borrowing a share that an idle tenant is not using.

The feature is **off by default**. With it off, or with fewer than two tenants
configured, the scheduler keeps its original queues and code path.

## Configuration

Tenant configuration is a JSON object passed to `--tenant-config` (online) or
`scheduler`/`vllm_config=...` (offline). Each tenant has a name, an optional
weight, and an optional running-request quota:

```bash
vllm serve meta-llama/Llama-3-8B-Instruct \
  --tenant-config '{
    "enabled": true,
    "tenants": [
      {"name": "team-a", "weight": 2.0},
      {"name": "team-b", "weight": 1.0, "max_running_reqs": 64},
      {"name": "team-c"}
    ],
    "share_tolerance": 0.05,
    "reclaim_max_steps": 8
  }'
```

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Turn on per-tenant fair sharing. |
| `tenants` | `[]` | Configured tenants. When non-empty, this is also the set of accepted identifiers. |
| `tenants[].name` | — | Identifier carried by requests. Must match `[A-Za-z0-9][A-Za-z0-9._:-]{0,63}` and be unique. |
| `tenants[].weight` | `1.0` | Relative share. Shares are proportional to weights among tenants that currently have work. |
| `tenants[].max_running_reqs` | none | Quota on concurrently running requests for this tenant. |
| `default_group` | `__default__` | Bucket for requests without an accepted tenant. Excluded from share accounting, so it behaves like the original queue. |
| `share_tolerance` | `0.05` | Maximum relative difference in served tokens between two active tenants. |
| `reclaim_max_steps` | `8` | Upper bound, in scheduling steps, on how long a tenant may stay over the share tolerance before a borrower is preempted to return its share. Also the reporting threshold for a missed reclaim. |
| `max_preemptions_per_step` | `1` | Cap on fairness-driven preemptions per step. |

## Declaring a tenant on a request

Tenant identity comes **only** from a validated identifier carried by the
request. It is never derived from the request id or from arrival order.

- **Online**: send the `X-Vllm-Tenant-Id` header. It applies to the
  Completions, Chat Completions, Responses, and generate APIs.

  ```python
  completion = client.chat.completions.create(
      model="meta-llama/Llama-3-8B-Instruct",
      messages=[{"role": "user", "content": "Hello!"}],
      extra_headers={"X-Vllm-Tenant-Id": "team-a"},
  )
  ```

- **Offline**: pass `tenant=` to `LLM.generate`, `LLM.chat`, `LLM.enqueue`, or
  `LLM.enqueue_chat`. A single value applies to every prompt; a list is paired
  one by one with the prompts.

  ```python
  llm.generate(prompts, sampling_params, tenant=["team-a", "team-b", "team-c"])
  ```

A request whose identifier is missing, malformed, or not in the configured
`tenants` list is served from the default group. That decision is logged once
per distinct identifier. Only the identifier and counts are logged; prompt
content and request bodies never appear in logs or metrics.

## How sharing works

1. **Shares.** Each active tenant's share is its weight divided by the total
   weight of the tenants that currently have work. Idle tenants are excluded,
   so their share is lent to the active tenants and throughput does not drop
   because of empty tenants.
2. **Slot budget.** The running-slot capacity (`max_num_seqs`) is divided among
   the active tenants in proportion to their shares. Each tenant gets the whole
   part of its share; leftover slots go to the tenants that have been served
   least so far, so the extra slots rotate over time. A tenant may not hold more
   running requests than its budget.
3. **Ordering.** When a slot frees, the scheduler admits the request of the
   tenant with the least *weighted service* (tokens served divided by the
   tenant's share). Within one tenant, the configured FCFS or priority order
   still decides.
4. **Borrowing.** A tenant is served beyond its own share while others are idle.
   Borrowing and lending are computed against the configured weights from one
   ledger, so the scheduling decision and the reported `borrowed`/`lent`
   counters always agree.
5. **Reclaim.** If a tenant with queued work falls behind the share tolerance
   (for example because a borrower holds its slots) for longer than
   `reclaim_max_steps`, the scheduler preempts a request from the tenant that is
   furthest ahead and hands the slot to the starved tenant. Preemption frees the
   request's KV cache, so a preempted request retains no cache and is recomputed
   from scratch when it resumes. If a starved tenant waits past the window, a
   warning naming the tenant and its counts is logged once.

### Fairness guarantee

Shares are made **exact** when the running-slot capacity divides evenly among
the active tenants (for example, 20 active tenants with `max_num_seqs` 20, 40,
or 60): each tenant receives the same number of running slots, so equal-share
tenants get equal tokens every step and the share difference is zero.

When the capacity does not divide evenly, tenancy fairness is limited by the
fact that slots are indivisible: a tenant can hold one slot more than another
at any instant. The scheduler rotates the leftover slots between the least
served tenants, so average per-tenant service still converges to the configured
shares, but the instantaneous difference can briefly exceed `share_tolerance`.
Set `share_tolerance` to the spread the deployment must actually hold, and size
`max_num_seqs` so that it divides evenly among the tenants you expect to be
active at once when you need the exact bound.

Note that when there are **more active tenants than running slots**, some
tenants cannot run at every instant; the scheduler time-slices slots among them
using the same reclaim mechanism, and their long-run service is proportional to
their shares but cannot be made exact at every step.

## Metrics

When the feature is enabled, the following metrics are reported. The tenant
label is the validated identifier.

| Metric | Type | Meaning |
| --- | --- | --- |
| `vllm:tenant_served_tokens` | Gauge | Cumulative tokens served to the tenant while it has been active. |
| `vllm:tenant_borrowed_tokens` | Gauge | Tokens the tenant was served beyond its configured share. |
| `vllm:tenant_lent_tokens` | Gauge | Tokens of the tenant's own share that borrowers used. |

The existing metric set is unchanged when the feature is off.

## Behavior notes

- The default group does not participate in cross-tenant share computation.
  A deployment with no declared tenants behaves exactly as before.
- Intra-tenant priority is unchanged: tenants are ordered against each other,
  and each tenant's own requests keep the configured policy.
- Turning the feature off, or configuring a single tenant, uses the original
  queues with no extra reorder or copy.
