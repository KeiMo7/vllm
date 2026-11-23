# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tenant weighted fair-share accounting for the "fairshare" policy.

The accountant is the single source of truth for everything the fairshare
policy needs beyond a plain queue:

- Virtual times for weighted-fair-queuing tenant selection.
- Per-tenant charging of computed tokens (or admitted requests), with a
  per-request high-water mark so tokens recomputed after a preemption are
  never charged twice.
- Bounded-wait and priority-aging inputs that are fully deterministic:
  every scheduling input derives from the request stream itself (arrival
  times, logical scheduling rounds), never from the wall clock, so the same
  arrival sequence always produces the same schedule.
- Per-window settlement records and a fairness report for reconciliation.

Scheduling decisions read only `virtual_time`, `is_under_reserved` and the
request stream; window settlement (`maybe_advance_window`) is accounting
only and never feeds back into scheduling.
"""

import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.config.scheduler import FairShareConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

DEFAULT_TENANT_ID = "default"

# Upper edges (seconds) of the wait-time histogram buckets; the last bucket
# collects everything above the final edge.
_WAIT_BUCKET_EDGES_S = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 60.0)


@dataclass
class WaitStats:
    """Waiting-time samples for one tenant over one accounting scope."""

    count: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0
    max_rounds: int = 0
    buckets: list[int] = field(
        default_factory=lambda: [0] * (len(_WAIT_BUCKET_EDGES_S) + 1)
    )

    def add(self, seconds: float, rounds: int) -> None:
        self.count += 1
        self.total_seconds += seconds
        self.max_seconds = max(self.max_seconds, seconds)
        self.max_rounds = max(self.max_rounds, rounds)
        for i, edge in enumerate(_WAIT_BUCKET_EDGES_S):
            if seconds <= edge:
                self.buckets[i] += 1
                return
        self.buckets[-1] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_seconds": (self.total_seconds / self.count if self.count else 0.0),
            "max_seconds": self.max_seconds,
            "max_rounds": self.max_rounds,
            "bucket_edges_seconds": list(_WAIT_BUCKET_EDGES_S),
            "buckets": list(self.buckets),
        }


@dataclass
class TenantCounters:
    """Usage counters for one tenant over one accounting scope."""

    tokens: int = 0
    requests_admitted: int = 0
    preemptions: int = 0
    wait: WaitStats = field(default_factory=WaitStats)

    def snapshot(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "requests_admitted": self.requests_admitted,
            "preemptions": self.preemptions,
            "wait": self.wait.snapshot(),
        }


class TenantAccountant:
    """Tracks per-tenant usage and virtual times for fair-share scheduling.

    Args:
        config: The fair-share configuration (already validated).
        time_fn: Clock used for wait-time and window accounting. Never used
            for scheduling decisions, so injecting a fake clock in tests does
            not affect replay determinism.

    """

    def __init__(
        self,
        config: FairShareConfig,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._time_fn = time_fn
        self._weights = dict(config.tenant_weights)
        self._virtual_time: dict[str, float] = {}
        # req_id -> (enqueue scheduling round, enqueue timestamp)
        self._enqueue_marks: dict[str, tuple[int, float]] = {}
        # Requests accounted under the default tenant for lack of an id.
        self.num_defaulted_requests = 0
        # Lifetime and current-window counters, per tenant.
        self._lifetime: dict[str, TenantCounters] = {}
        self._current: dict[str, TenantCounters] = {}
        now = self._time_fn()
        self._window_start = now
        self.windows: list[dict[str, Any]] = []
        # Deterministic sliding window of recent charges, for reservations.
        self._recent: deque[tuple[str, float]] = deque()
        self._recent_by_tenant: dict[str, float] = {}
        self._recent_total = 0.0
        # Charges applied in the current scheduling step, for undoing the
        # charge of a request that is preempted later in the same step:
        # req_id -> (charged delta, high-water mark before the charge, tenant)
        self._step_charges: dict[str, tuple[int, int | None, str]] = {}

    # -- identity ---------------------------------------------------------

    def tenant_of(self, request: "Request") -> str:
        """Trusted tenant of a request; missing ids go to the default tenant."""
        return request.tenant_id or DEFAULT_TENANT_ID

    def weight_of(self, tenant: str) -> float:
        return self._weights.get(tenant, self._config.default_tenant_weight)

    # -- scheduling inputs (deterministic) --------------------------------

    @property
    def priority_aging_seconds(self) -> float:
        return self._config.priority_aging_seconds

    def virtual_time(self, tenant: str) -> float:
        return self._virtual_time.get(tenant, 0.0)

    def is_under_reserved(self, tenant: str) -> bool:
        """Whether the tenant's recent usage is below its reserved share."""
        min_share = self._config.tenant_min_shares.get(tenant)
        if not min_share or self._recent_total <= 0:
            return False
        used = self._recent_by_tenant.get(tenant, 0.0)
        return used < min_share * self._recent_total

    def pick_victim(self, running: "list[Request]") -> "Request":
        """Pick the preemption victim among running requests.

        The victim comes from the tenant that is most over its share
        (highest virtual time); within that tenant, the most recently
        admitted request, mirroring the FCFS youngest-first choice.
        """
        assert running, "pick_victim requires a non-empty running list"
        over = max(self.virtual_time(self.tenant_of(r)) for r in running)
        for request in reversed(running):
            if self.virtual_time(self.tenant_of(request)) == over:
                return request
        raise AssertionError("unreachable")

    # -- charging ----------------------------------------------------------

    def begin_step(self, now: float | None = None) -> None:
        """Start a scheduling step: settle windows, drop stale undo state."""
        self._step_charges.clear()
        self.maybe_advance_window(now)

    def charge_scheduled_tokens(
        self, request: "Request", num_scheduled: int, computed_after: int
    ) -> None:
        """Charge a request for tokens scheduled in this step.

        A per-request high-water mark ensures each token position is charged
        at most once: tokens recomputed after a preemption (or replayed for
        any other reason) are not counted again, and prefix-cache hits are
        never charged because the first charge covers exactly the tokens
        scheduled in that step.

        Args:
            request: The request just scheduled.
            num_scheduled: Tokens scheduled for it in this step.
            computed_after: Its `num_computed_tokens` after this step's
                tokens are counted (the scheduler advances the field only
                after the whole step, so callers pass the sum explicitly).

        """
        if num_scheduled <= 0 or self._config.share_unit != "token":
            return
        hwm = request.fair_share_charged_upto
        delta = num_scheduled if hwm is None else max(0, computed_after - hwm)
        request.fair_share_charged_upto = max(computed_after, hwm or 0)
        if delta <= 0:
            return
        tenant = self.tenant_of(request)
        self._charge(tenant, delta, tokens=delta)
        self._step_charges[request.request_id] = (delta, hwm, tenant)

    def undo_charge(self, request: "Request") -> None:
        """Reverse this step's charge of a request preempted before it ran.

        Only valid for a request charged earlier in the same scheduling
        step; the reversal is appended to the recent-usage window as a
        negative entry so the window sum stays exact.
        """
        entry = self._step_charges.pop(request.request_id, None)
        if entry is None:
            return
        delta, hwm_before, tenant = entry
        self._charge(tenant, -delta, tokens=-delta)
        request.fair_share_charged_upto = hwm_before

    def _charge(self, tenant: str, cost: float, tokens: int = 0) -> None:
        self._virtual_time[tenant] = self.virtual_time(tenant) + cost / self.weight_of(
            tenant
        )
        if tokens:
            self._counters(self._lifetime, tenant).tokens += tokens
            self._counters(self._current, tenant).tokens += tokens
        self._recent.append((tenant, cost))
        self._recent_by_tenant[tenant] = self._recent_by_tenant.get(tenant, 0.0) + cost
        self._recent_total += cost
        limit = self._config.reservation_window_tokens
        while self._recent_total > limit and self._recent:
            old_tenant, old_cost = self._recent.popleft()
            self._recent_total -= old_cost
            remaining = self._recent_by_tenant[old_tenant] - old_cost
            if remaining > 0:
                self._recent_by_tenant[old_tenant] = remaining
            else:
                del self._recent_by_tenant[old_tenant]

    # -- request lifecycle hooks -------------------------------------------

    def note_new_request(self, request: "Request", round_: int) -> None:
        """Account the arrival of a genuinely new request."""
        if not request.tenant_id:
            self.num_defaulted_requests += 1
        self.note_enqueue(request, round_)

    def note_enqueue(self, request: "Request", round_: int) -> None:
        """Mark when a request (re-)entered a waiting queue."""
        self._enqueue_marks[request.request_id] = (round_, self._time_fn())

    def note_dequeue(self, request_id: str) -> None:
        """Drop the enqueue mark of a request removed while waiting."""
        self._enqueue_marks.pop(request_id, None)

    def note_admission(self, request: "Request", round_: int) -> None:
        """Account a waiting request becoming running."""
        tenant = self.tenant_of(request)
        mark = self._enqueue_marks.pop(request.request_id, None)
        if mark is not None:
            enqueue_round, enqueue_ts = mark
            wait_seconds = max(0.0, self._time_fn() - enqueue_ts)
            wait_rounds = max(0, round_ - enqueue_round)
            self._counters(self._lifetime, tenant).wait.add(wait_seconds, wait_rounds)
            self._counters(self._current, tenant).wait.add(wait_seconds, wait_rounds)
        self._counters(self._lifetime, tenant).requests_admitted += 1
        self._counters(self._current, tenant).requests_admitted += 1
        if self._config.share_unit == "request":
            self._charge(tenant, 1.0)

    def note_preemption(self, request: "Request") -> None:
        tenant = self.tenant_of(request)
        self._counters(self._lifetime, tenant).preemptions += 1
        self._counters(self._current, tenant).preemptions += 1

    @staticmethod
    def _counters(store: dict[str, TenantCounters], tenant: str) -> TenantCounters:
        counters = store.get(tenant)
        if counters is None:
            counters = store[tenant] = TenantCounters()
        return counters

    # -- window settlement ---------------------------------------------------

    def maybe_advance_window(self, now: float | None = None) -> None:
        """Settle elapsed accounting windows; usage never carries over."""
        if now is None:
            now = self._time_fn()
        size = self._config.window_size_seconds
        while now >= self._window_start + size:
            self.windows.append(
                {
                    "start": self._window_start,
                    "end": self._window_start + size,
                    "tenants": {
                        tenant: counters.snapshot()
                        for tenant, counters in sorted(self._current.items())
                    },
                }
            )
            self._current = {}
            self._window_start += size

    # -- reporting -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Live per-tenant stats for `SchedulerStats.fair_share_stats`."""
        self.maybe_advance_window()
        return {
            "share_unit": self._config.share_unit,
            "num_defaulted_requests": self.num_defaulted_requests,
            "tenants": {
                tenant: {
                    "weight": self.weight_of(tenant),
                    "virtual_time": self.virtual_time(tenant),
                    **self._counters(self._lifetime, tenant).snapshot(),
                }
                for tenant in sorted(self._known_tenants())
            },
        }

    def fairness_report(self) -> dict[str, Any]:
        """Per-tenant fairness report: share deviation and wait distribution."""
        self.maybe_advance_window()
        tenants = sorted(self._known_tenants())
        by_token = self._config.share_unit == "token"

        def measure(tenant: str) -> int:
            counters = self._counters(self._lifetime, tenant)
            return counters.tokens if by_token else counters.requests_admitted

        total = sum(measure(t) for t in tenants)
        active = [t for t in tenants if measure(t) > 0]
        active_weight = sum(self.weight_of(t) for t in active)
        per_tenant: dict[str, Any] = {}
        for tenant in tenants:
            counters = self._counters(self._lifetime, tenant)
            actual = measure(tenant) / total if total else 0.0
            expected = (
                self.weight_of(tenant) / active_weight if tenant in active else 0.0
            )
            per_tenant[tenant] = {
                "weight": self.weight_of(tenant),
                "expected_share": expected,
                "actual_share": actual,
                "share_deviation": actual - expected,
                **counters.snapshot(),
            }
        return {
            "share_unit": self._config.share_unit,
            "window_size_seconds": self._config.window_size_seconds,
            "num_defaulted_requests": self.num_defaulted_requests,
            "total_charged": total,
            "tenants": per_tenant,
            "windows": self.windows,
        }

    def export_report(self, report_dir: str | Path) -> Path:
        """Write the fairness report and per-window settlement detail to disk."""
        self.maybe_advance_window()
        out_dir = Path(report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = self.fairness_report()
        windows = report.pop("windows")
        report_path = out_dir / "fairness_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        with (out_dir / "fair_share_windows.jsonl").open("w") as f:
            for window in windows:
                f.write(json.dumps(window) + "\n")
        logger.info("Fair-share report written to %s", report_path)
        return report_path

    def _known_tenants(self) -> set[str]:
        return (
            set(self._lifetime)
            | set(self._current)
            | set(self._virtual_time)
            | set(self._weights)
        )
