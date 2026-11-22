# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tenant-share ledger and tenant-aware waiting queue.

Plain Python with no engine or GPU dependency, so the fairness arithmetic can
be tested on its own.

Two kinds of accounting live here, and both are read back by the scheduler and
by the metrics reporter, so a scheduling decision and the reported numbers
cannot disagree:

- *Cumulative* service per tenant, used to report borrowing and lending
  against the configured weights (an idle tenant's share that a borrower uses
  shows up as `borrowed`).
- *Epoch* weighted service, used to order tenants. An epoch restarts whenever
  the set of declared tenants with work changes. Restarting is what bounds
  borrowing: a tenant that was alone cannot accumulate an unbounded credit and
  then lock the cluster out when a lender returns.

The default group is queued and served like any other tenant but is excluded
from share, borrowing, and ordering arithmetic, so a deployment that declares
no tenants keeps the original behavior.
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from vllm.config.tenant import TenantConfig
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request


@dataclass
class TenantLedger:
    """Single source of truth for shares, borrowing, and reclamation."""

    config: TenantConfig
    # Cumulative tokens served per declared tenant since it became resident.
    _cum_served: dict[str, int] = field(default_factory=dict)
    # Epoch service, in weighted tokens (tokens / normalized share).
    _epoch_weighted: dict[str, float] = field(default_factory=dict)
    # Tokens scheduled in the current step, folded in by `commit_step`.
    _pending: dict[str, int] = field(default_factory=dict)
    # Declared tenants with queued or running work.
    _active: set[str] = field(default_factory=set)
    current_step: int = 0
    epoch: int = 0
    _version: int = 0
    # Consecutive steps each tenant has been over the share tolerance while
    # unable to run; drives the bounded reclaim grace and its report.
    _starved_steps: dict[str, int] = field(default_factory=dict)
    # Cached normalized shares, keyed on the active set. Recomputing this on
    # every score lookup would make ranking a step O(n^2) in the tenant count.
    _shares_cache: dict[str, float] = field(default_factory=dict)
    _shares_cache_valid: bool = False

    def __post_init__(self) -> None:
        self._default = self.config.default_group
        self._configured_weight = (
            sum(share.weight for share in self.config.tenants) or 1.0
        )

    # -- lifecycle --------------------------------------------------------
    def reset(self, step: int = 0) -> None:
        self._cum_served.clear()
        self._epoch_weighted.clear()
        self._pending.clear()
        self._active.clear()
        self._starved_steps.clear()
        self.current_step = step
        self.epoch = 0
        self._invalidate_shares()
        self._version += 1

    def begin_step(self, step: int) -> None:
        self.current_step = step

    def set_active_tenants(self, tenants: Iterable[str]) -> None:
        """Replace the declared tenants that currently have work.

        An epoch ends whenever the active set changes. Restarting the epoch
        clears per-epoch service, which bounds borrowing: no tenant carries a
        credit large enough to monopolize the cluster when another returns.
        Cumulative lifetime counters are left alone.
        """
        new_active = {t for t in tenants if t != self._default}
        if new_active == self._active:
            return
        self._active = new_active
        self._invalidate_shares()
        # Tenants that are no longer active stop being starved.
        self._starved_steps = {
            t: n for t, n in self._starved_steps.items() if t in new_active
        }

        self.epoch += 1
        self._epoch_weighted.clear()
        self._pending.clear()
        self._version += 1

    def release_tenant(self, tenant: str) -> None:
        """Release a tenant that has no requests left.

        Cached KV and running slots are freed by the scheduler independently.
        Here the tenant simply stops taking part in shares and drops its
        per-epoch ordering state. Its cumulative served count is kept: it is a
        lifetime counter for the `borrowed`/`lent` metrics, and resetting it
        would punish a tenant that briefly drains its queue (for example
        between two of its own requests) by making it re-earn service from
        zero while the others are far ahead.
        """
        if tenant == self._default:
            return
        self._epoch_weighted.pop(tenant, None)
        self._pending.pop(tenant, None)
        self._active.discard(tenant)
        self._starved_steps.pop(tenant, None)
        self._invalidate_shares()
        self._version += 1

    # -- activity ---------------------------------------------------------
    def is_share_participant(self, tenant: str) -> bool:
        return tenant != self._default and tenant in self._active

    def active_tenants(self) -> set[str]:
        return set(self._active)

    # -- shares -----------------------------------------------------------
    def shares(self) -> dict[str, float]:
        """Normalized share over the tenants that currently have work.

        Idle tenants are excluded, so their share is lent to the active ones
        and throughput does not drop because of empty tenants.
        """
        if self._shares_cache_valid:
            return self._shares_cache
        if not self._active:
            shares: dict[str, float] = {}
        else:
            total = sum(self.config.weight_of(t) for t in self._active)
            shares = {t: self.config.weight_of(t) / total for t in self._active}
        self._shares_cache = shares
        self._shares_cache_valid = True
        return shares

    def _invalidate_shares(self) -> None:
        self._shares_cache_valid = False

    def target_fraction(self, tenant: str) -> float:
        if tenant == self._default:
            return 0.0
        return self.shares().get(tenant, 0.0)

    def configured_fraction(self, tenant: str) -> float:
        """Share of the whole configured tenant set, whether active or not."""
        if tenant == self._default:
            return 0.0
        return self.config.weight_of(tenant) / self._configured_weight

    def slot_budget(self, capacity: int) -> dict[str, int]:
        """Whole running slots assigned to each active tenant.

        Slot targets are fractional in general (`share * capacity`), but slots
        are indivisible. Each tenant gets the whole part of its target, and the
        leftover slots go to the tenants that have been served least over their
        lifetime. Because a tenant holding an extra slot pulls ahead of the
        others, the extra slot keeps moving to a different tenant, so average
        slot counts converge to the fractional targets. Rotation is driven by
        cumulative rather than per-epoch service, which keeps it monotone and
        avoids oscillation.
        """
        shares = self.shares()
        if not shares:
            return {}
        raw = {t: s * capacity for t, s in shares.items()}
        budget = {t: int(v) for t, v in raw.items()}
        leftover = capacity - sum(budget.values())
        if leftover <= 0:
            return budget
        # Least lifetime-served first; among equals, the larger fractional part
        # first, then the name for determinism.
        order = sorted(
            shares, key=lambda t: (self._cum_served.get(t, 0), -(raw[t] % 1.0), t)
        )
        for tenant in order[:leftover]:
            budget[tenant] += 1
        return budget

    # -- service ----------------------------------------------------------
    def record_pending(self, tenant: str, num_tokens: int) -> None:
        """Account tokens scheduled this step for `tenant`.

        `num_tokens` may be negative to give back service for a request that
        was scheduled and then preempted within the same step.
        """
        if tenant == self._default:
            return
        total = self._pending.get(tenant, 0) + num_tokens
        if total > 0:
            self._pending[tenant] = total
        else:
            self._pending.pop(tenant, None)

    def commit_step(self) -> None:
        for tenant, tokens in self._pending.items():
            self._cum_served[tenant] = self._cum_served.get(tenant, 0) + tokens
            share = max(self.target_fraction(tenant), 1e-12)
            self._epoch_weighted[tenant] = (
                self._epoch_weighted.get(tenant, 0.0) + tokens / share
            )
            self._version += 1
        self._pending.clear()

    def served(self, tenant: str) -> int:
        """Cumulative tokens served to `tenant` since it became resident."""
        return self._cum_served.get(tenant, 0)

    def weighted_served(self, tenant: str) -> float:
        if tenant == self._default:
            return 0.0
        weight = self._epoch_weighted.get(tenant, 0.0)
        pending = self._pending.get(tenant, 0)
        if pending:
            weight += pending / max(self.target_fraction(tenant), 1e-12)
        return weight

    def score(self, tenant: str) -> float:
        """Sort key for the next admission: least weighted service first."""
        if tenant == self._default:
            # The default group never takes share from a declared tenant.
            return float("inf")
        return self.weighted_served(tenant)

    def rank(self, tenants: Iterable[str]) -> list[str]:
        return sorted(tenants, key=lambda t: (self.score(t), t))

    # -- borrowing --------------------------------------------------------
    def total_served(self) -> int:
        return sum(self._cum_served.get(t, 0) for t in self._active)

    def borrowed(self, tenant: str) -> int:
        """Cumulative tokens served beyond the configured share."""
        if not self.is_share_participant(tenant):
            return 0
        total = self.total_served()
        if total <= 0:
            return 0
        fair = self.configured_fraction(tenant) * total
        return max(0, int(self.served(tenant) - fair))

    def lent(self, tenant: str) -> int:
        """Cumulative tokens of this tenant's share used by borrowers."""
        if not self.is_share_participant(tenant):
            return 0
        total = self.total_served()
        if total <= 0:
            return 0
        fair = self.configured_fraction(tenant) * total
        return max(0, int(fair - self.served(tenant)))

    # -- starvation -------------------------------------------------------
    def starving_tenant(
        self, running: dict[str, int], queued: dict[str, int]
    ) -> str | None:
        """The tenant most owed service that cannot run, or `None`.

        Weighted service is the invariant the scheduler keeps within
        `share_tolerance`, so starvation is measured against it rather than
        against slot counts: with more tenants than running slots, a tenant's
        target can be below one slot, and only time-slicing (preempting a
        borrower so the starved tenant gets a slot for a while, then rotating)
        keeps service fair. A tenant qualifies only if it has queued work the
        scheduler could actually run if a slot freed.
        """
        active = self.active_tenants()
        self._starved_steps = {
            t: n for t, n in self._starved_steps.items() if t in active
        }
        if len(active) < 2:
            return None
        served = {t: self.weighted_served(t) for t in active}
        mean = sum(served.values()) / len(served)
        if mean <= 0:
            return None
        worst = min(active, key=lambda t: (served[t], t))
        over_tolerance = (max(served.values()) - served[worst]) / mean > (
            self.config.share_tolerance
        )
        if not over_tolerance or not queued.get(worst, 0):
            self._starved_steps.pop(worst, None)
            return None
        self._starved_steps[worst] = self._starved_steps.get(worst, 0) + 1
        return worst

    def starvation_steps(self, tenant: str) -> int:
        """Consecutive steps `tenant` has been starved (for observability)."""
        return self._starved_steps.get(tenant, 0)

    def starvation_overdue(self, tenant: str) -> bool:
        """Whether a starved tenant has waited past `reclaim_max_steps`."""
        return self.starvation_steps(tenant) > self.config.reclaim_max_steps

    def borrower_to_preempt(self, running: dict[str, int]) -> str | None:
        """The tenant holding the most running slots above its target."""
        active = self.active_tenants()
        candidates = [t for t in active if running.get(t, 0) > 0]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda t: (self.weighted_served(t), running.get(t, 0), t),
        )

    def unfairness(self) -> float:
        """Relative spread of weighted service across active tenants."""
        if len(self._active) < 2:
            return 0.0
        values = [self.weighted_served(t) for t in self._active]
        mean = sum(values) / len(values)
        if mean <= 0:
            return 0.0
        return (max(values) - min(values)) / mean

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Per-tenant counters for metrics: only identifiers and counts."""
        out: dict[str, dict[str, int]] = {}
        for tenant in sorted(self._active):
            out[tenant] = {
                "served": self.served(tenant),
                "borrowed": self.borrowed(tenant),
                "lent": self.lent(tenant),
            }
        return out

    @property
    def version(self) -> int:
        return self._version


class TenantRequestQueue(RequestQueue):
    """One underlying `RequestQueue` per tenant, visited in ledger order.

    Requests keep the scheduler's configured policy (FCFS or priority) inside
    their own tenant's queue, so intra-tenant priority is unchanged; the
    tenant layer only decides which tenant is asked for its next request.

    The tenant order is rebuilt at most once per step (`refresh`), so a step's
    overhead is `O(n log n)` in the number of tenants with queued work, not
    per request.
    """

    def __init__(
        self,
        policy_factory: Callable[[], RequestQueue],
        ledger: TenantLedger,
    ) -> None:
        self._new_queue = policy_factory
        self._ledger = ledger
        self._queues: dict[str, RequestQueue] = {}
        self._order: list[str] = []
        self._rebuild_version = -1
        self._size = 0

    # -- internal ---------------------------------------------------------
    def _queue(self, tenant: str) -> RequestQueue:
        queue = self._queues.get(tenant)
        if queue is None:
            queue = self._new_queue()
            self._queues[tenant] = queue
        return queue

    def _invalidate(self) -> None:
        """Force the next `refresh` to rebuild the visit order.

        Membership changes (a queue becoming empty or non-empty) must
        invalidate even when no score changed, otherwise a tenant that just
        received its first request could be missing from the order and its
        request stranded behind the default group.
        """
        self._rebuild_version = -1

    def refresh(self) -> None:
        """Recompute the visit order if tenant scores or membership changed."""
        if self._rebuild_version == self._ledger.version:
            return
        self._rebuild_version = self._ledger.version
        # Declared tenants first, least served first; the default group last.
        declared = [
            tenant
            for tenant, queue in self._queues.items()
            if queue and tenant != self._ledger.config.default_group
        ]
        self._order = self._ledger.rank(declared)

    def _first_nonempty(self) -> RequestQueue | None:
        self.refresh()
        for tenant in self._order:
            queue = self._queues.get(tenant)
            if queue:
                return queue
        default = self._ledger.config.default_group
        queue = self._queues.get(default)
        return queue if queue else None

    # -- RequestQueue -----------------------------------------------------
    def add_request(self, request: Request) -> None:
        self._queue(request.tenant_id).add_request(request)
        self._size += 1
        self._invalidate()

    def pop_request(self) -> Request:
        queue = self._first_nonempty()
        if queue is None:
            raise IndexError("pop from empty queue")
        self._size -= 1
        self._invalidate()
        return queue.pop_request()

    def peek_request(self) -> Request:
        queue = self._first_nonempty()
        if queue is None:
            raise IndexError("peek from an empty queue")
        return queue.peek_request()

    def prepend_request(self, request: Request) -> None:
        self._queue(request.tenant_id).prepend_request(request)
        self._size += 1
        self._invalidate()

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        queue = self._queues.get(request.tenant_id)
        if queue is None:
            raise ValueError("request is not in the queue")
        queue.remove_request(request)
        self._size -= 1
        self._invalidate()

    def remove_requests(self, requests: Iterable[Request]) -> None:
        # Group by tenant so each underlying queue removes in one pass; the
        # priority queue's bulk remove is far cheaper than a membership test
        # and removal per request.
        by_tenant: dict[str, list[Request]] = {}
        for request in requests:
            by_tenant.setdefault(request.tenant_id, []).append(request)
        for tenant, group in by_tenant.items():
            queue = self._queues.get(tenant)
            if queue is None or not queue:
                continue
            before = len(queue)
            queue.remove_requests(group)
            self._size -= before - len(queue)
            self._invalidate()

    def __bool__(self) -> bool:
        return self._size > 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Request]:
        self.refresh()
        for tenant in self._order:
            queue = self._queues.get(tenant)
            if queue:
                yield from queue
        default = self._ledger.config.default_group
        queue = self._queues.get(default)
        if queue:
            yield from queue

    # -- tenant helpers ---------------------------------------------------
    def tenant_len(self, tenant: str) -> int:
        queue = self._queues.get(tenant)
        return len(queue) if queue else 0

    def queued_tenants(self) -> set[str]:
        return {t for t, q in self._queues.items() if q}

    def drop_empty(self) -> set[str]:
        """Forget empty per-tenant queues; return the tenants dropped."""
        dropped = {
            tenant
            for tenant, queue in self._queues.items()
            if not queue and tenant != self._ledger.config.default_group
        }
        for tenant in dropped:
            del self._queues[tenant]
        return dropped


def make_tenant_queue(
    policy_factory: Callable[[], RequestQueue], ledger: TenantLedger
) -> TenantRequestQueue:
    return TenantRequestQueue(policy_factory, ledger)
