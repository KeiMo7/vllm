# Fair-Share Scheduling

When several business tenants share one serving cluster, arrival-order
(FCFS) scheduling lets a burst from one tenant starve the others, and a
static priority only moves the problem: raise your priority and you
preempt everyone, lower it and you queue forever. The `fairshare`
scheduling policy instead divides compute across tenants in proportion to
configured weights, with a bounded worst-case wait for every tenant.

Enable it with the existing policy switch and a `--fair-share` config:

```bash
vllm serve <model> \
  --scheduling-policy fairshare \
  --fair-share '{"tenant_weights": {"team_a": 2.0, "team_b": 1.0}}'
```

When the policy is not `fairshare`, the `fair_share` section is ignored
and FCFS/priority behave exactly as before.

## Tenant identity

The scheduler never infers tenancy from request content. The serving
layer reads the tenant from the `X-Vllm-Tenant-Id` HTTP header, which
your authenticated gateway sets after identifying the caller; the value
travels on the request as a trusted `tenant_id` down to the scheduler.
Requests without a tenant id are accounted under a reserved `default`
tenant (weighted by `default_tenant_weight`) and counted in
`num_defaulted_requests`, so a missing identity is visible in the
metrics rather than silently passed through.

## How shares are enforced

Fairness is enforced inside the scheduling loop, not by a single sort
key:

- **Admission.** Each waiting queue is one FIFO per tenant. The next
  request admitted is taken from the tenant with the lowest virtual
  time (weighted-fair-queuing): every token a tenant is charged advances
  its virtual time by `tokens / weight`, so over-served tenants yield to
  under-served ones at every admission, including multiple admissions
  within one engine step.
- **Preemption.** Preemption still only happens when a running request
  cannot allocate KV blocks. The victim is chosen from the tenant most
  over its share (highest virtual time), most recently admitted first,
  so reclaimed capacity flows back to under-served tenants. A preempted
  request resumes later; a per-request high-water mark ensures
  recomputed tokens are never charged to its tenant twice.
- **Priorities.** `Request.priority` still works: requests in a better
  priority band go first, but bands age (`priority_aging_seconds`), so a
  low-priority request cannot be delayed indefinitely.

Because virtual times equalize served compute per unit weight, a
backlogged tenant's worst wait is bounded by roughly one batch turnover
regardless of how large the other tenants' bursts are.

## Configuration

All fields live under [FairShareConfig][vllm.config.scheduler.FairShareConfig]
and are validated at startup when the policy is `fairshare`; a
non-positive weight or an over-subscribed reservation refuses to start
and names the offending tenant.

- `tenant_weights`: per-tenant weight. Shares of compute follow these
  proportions among tenants with outstanding demand.
- `share_unit`: `"token"` (default) charges per computed token, so long
  requests cost more share; `"request"` charges once per admitted
  request, so every request costs the same regardless of length.
- `tenant_min_shares`: reserved fractions of capacity per tenant. A
  backlogged tenant below its reservation is admitted first; idle
  reservations are borrowed by the other tenants.
- `window_size_seconds`: length of the accounting window used for
  settlement. Usage is settled per window and never carried across
  windows.
- `priority_aging_seconds`: how often a waiting request's effective
  priority improves by one band.

Scheduling decisions depend only on the request stream (arrival times,
weights, logical rounds), never on the wall clock, so the same arrival
sequence always replays to the identical schedule.

## Observability and reconciliation

Per-tenant counters (charged tokens, admitted requests, preemptions,
waiting-time distribution) are exported in `SchedulerStats` as
`fair_share_stats`, alongside the unchanged existing fields. For
end-of-month reconciliation, `TenantAccountant.fairness_report()`
computes each tenant's expected vs. actual share and its deviation, and
`TenantAccountant.export_report(dir)` writes `fairness_report.json`
plus a per-window settlement detail `fair_share_windows.jsonl` that can
be archived per billing period.
