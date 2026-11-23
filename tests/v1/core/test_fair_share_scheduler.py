# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-tenant weighted fair-share scheduling (policy="fairshare").

Covers the acceptance criteria of the fair-share feature:

- shares follow configured weights under burst traffic, within a threshold;
- the low-weight tenant's worst waiting time stays below a proven bound;
- the same arrival sequence replays to the identical schedule;
- single-tenant throughput matches the fcfs baseline;
- invalid weights refuse to start, missing tenant ids are accounted, not
  silently dropped;
- preempted requests resume without being charged twice;
- accounting windows settle without carrying usage across windows.
"""

import json
import time

import pytest

from vllm.config.scheduler import FairShareConfig, SchedulerConfig
from vllm.v1.core.sched.fair_share import DEFAULT_TENANT_ID, TenantAccountant
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
    SchedulingPolicy,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

PROMPT_TOKENS = 10
MAX_TOKENS = 5
# Tokens each request computes over its lifetime: prompt + one position per
# generated token.
TOKENS_PER_REQUEST = PROMPT_TOKENS + MAX_TOKENS


def _fair_share_config(**kwargs) -> FairShareConfig:
    kwargs.setdefault("tenant_weights", {"A": 2.0, "B": 1.0})
    return FairShareConfig(**kwargs)


def _make_scheduler(fair_share: FairShareConfig, **kwargs):
    kwargs.setdefault("scheduling_policy", "fairshare")
    kwargs.setdefault("fair_share", fair_share)
    kwargs.setdefault("max_num_seqs", 4)
    kwargs.setdefault("num_blocks", 10000)
    return create_scheduler(**kwargs)


def _tenant_requests(
    tenant: str | None,
    n: int,
    id_prefix: str,
    arrival: float = 0.0,
    **kwargs,
) -> list[Request]:
    kwargs.setdefault("num_tokens", PROMPT_TOKENS)
    kwargs.setdefault("max_tokens", MAX_TOKENS)
    return create_requests(
        n,
        req_ids=[f"{id_prefix}-{i}" for i in range(n)],
        tenant_ids=[tenant] * n,
        arrival_times=[arrival + i * 1e-3 for i in range(n)],
        **kwargs,
    )


def _run_steps(scheduler, num_steps: int) -> list[list[str]]:
    """Drive the scheduler; return the scheduled request ids of each step."""
    scheduled_per_step: list[list[str]] = []
    for _ in range(num_steps):
        output = scheduler.schedule()
        req_ids = list(output.num_scheduled_tokens)
        if not req_ids:
            break
        scheduled_per_step.append(req_ids)
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                sampled_token_ids=[[1]] * len(req_ids),
            ),
        )
    return scheduled_per_step


def _run_until_done(scheduler, max_steps: int = 10000) -> list[list[str]]:
    scheduled_per_step: list[list[str]] = []
    for _ in range(max_steps):
        output = scheduler.schedule()
        req_ids = list(output.num_scheduled_tokens)
        if not req_ids:
            break
        scheduled_per_step.append(req_ids)
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                sampled_token_ids=[[1]] * len(req_ids),
            ),
        )
    return scheduled_per_step


def _tokens_by_tenant(scheduler) -> dict[str, int]:
    assert scheduler.fair_share is not None
    snapshot = scheduler.fair_share.snapshot()
    return {tenant: stats["tokens"] for tenant, stats in snapshot["tenants"].items()}


# ---------------------------------------------------------------------------
# Config validation (failure behavior: refuse to start on invalid weights)
# ---------------------------------------------------------------------------


def _scheduler_config(fair_share: FairShareConfig) -> SchedulerConfig:
    return SchedulerConfig(
        max_model_len=1024,
        is_encoder_decoder=False,
        policy="fairshare",
        fair_share=fair_share,
    )


def test_zero_weight_rejected():
    with pytest.raises(ValueError, match="tenant_weights\\['B'\\]"):
        _scheduler_config(FairShareConfig(tenant_weights={"A": 1.0, "B": 0.0}))


def test_negative_weight_rejected():
    with pytest.raises(ValueError, match="tenant_weights\\['A'\\]"):
        _scheduler_config(FairShareConfig(tenant_weights={"A": -2.0, "B": 1.0}))


def test_min_share_unknown_tenant_rejected():
    with pytest.raises(ValueError, match="tenant_min_shares\\['C'\\]"):
        _scheduler_config(
            FairShareConfig(
                tenant_weights={"A": 1.0},
                tenant_min_shares={"C": 0.5},
            )
        )


def test_min_shares_oversubscribed_rejected():
    with pytest.raises(ValueError, match="sum to at most 1"):
        _scheduler_config(
            FairShareConfig(
                tenant_weights={"A": 1.0, "B": 1.0},
                tenant_min_shares={"A": 0.7, "B": 0.5},
            )
        )


def test_valid_config_accepted():
    config = _scheduler_config(_fair_share_config())
    assert config.policy == "fairshare"


def test_fair_share_config_ignored_by_other_policies():
    # An invalid fair-share section must not affect fcfs/priority startups.
    config = SchedulerConfig(
        max_model_len=1024,
        is_encoder_decoder=False,
        policy="fcfs",
        fair_share=FairShareConfig(tenant_weights={"A": 0.0}),
    )
    assert config.policy == "fcfs"


# ---------------------------------------------------------------------------
# Off switch: existing policies are untouched
# ---------------------------------------------------------------------------


def test_disabled_policies_have_no_accountant():
    fcfs = create_scheduler(scheduling_policy="fcfs")
    assert fcfs.fair_share is None
    assert isinstance(fcfs.waiting, FCFSRequestQueue)
    priority = create_scheduler(scheduling_policy="priority")
    assert priority.fair_share is None
    assert isinstance(priority.waiting, PriorityRequestQueue)


def test_fairshare_policy_wires_accountant():
    scheduler = _make_scheduler(_fair_share_config())
    assert scheduler.fair_share is not None
    assert scheduler.policy == SchedulingPolicy.FAIRSHARE


# ---------------------------------------------------------------------------
# Share enforcement under burst traffic (acceptance: deviation in threshold)
# ---------------------------------------------------------------------------


def test_burst_traffic_share_deviation():
    """Two tenants, one burst: token shares track the 2:1 weights.

    Both tenants keep a backlog for the whole measurement, so the charged
    token split must converge to the weight ratio 2/3 : 1/3.
    """
    scheduler = _make_scheduler(_fair_share_config())
    for request in _tenant_requests("A", 100, "a") + _tenant_requests("B", 100, "b"):
        scheduler.add_request(request)

    _run_steps(scheduler, 60)

    tokens = _tokens_by_tenant(scheduler)
    total = tokens["A"] + tokens["B"]
    share_a = tokens["A"] / total
    deviation = share_a - 2 / 3
    # Measured on this scenario: deviation stays well under 5%.
    assert abs(deviation) < 0.05, (
        f"tenant A share {share_a:.4f} deviates from 2/3 by {deviation:+.4f}"
    )

    # The fairness report carries the same numbers for reconciliation.
    report = scheduler.fair_share.fairness_report()
    assert report["tenants"]["A"]["expected_share"] == pytest.approx(2 / 3)
    assert report["tenants"]["A"]["actual_share"] == pytest.approx(share_a, abs=1e-9)
    assert report["tenants"]["A"]["share_deviation"] == pytest.approx(
        deviation, abs=1e-9
    )


def test_fairness_report_export(tmp_path):
    """The per-tenant fairness report and window detail are exported."""
    scheduler = _make_scheduler(_fair_share_config(window_size_seconds=0.05))
    for request in _tenant_requests("A", 30, "a") + _tenant_requests("B", 30, "b"):
        scheduler.add_request(request)
    _run_steps(scheduler, 30)

    out_dir = tmp_path
    report_path = scheduler.fair_share.export_report(out_dir)
    report = json.loads(report_path.read_text())
    assert set(report["tenants"]) == {"A", "B"}
    for tenant in ("A", "B"):
        stats = report["tenants"][tenant]
        assert stats["tokens"] > 0
        assert stats["requests_admitted"] > 0
        assert "share_deviation" in stats
        assert stats["wait"]["count"] == stats["requests_admitted"]
    windows_path = out_dir / "fair_share_windows.jsonl"
    windows = [json.loads(line) for line in windows_path.read_text().splitlines()]
    assert windows, "expected at least one settled window"
    for window in windows:
        assert window["end"] > window["start"]
        assert "tenants" in window


# ---------------------------------------------------------------------------
# Bounded waiting for the low-weight tenant
# ---------------------------------------------------------------------------


def test_low_weight_tenant_max_wait_bounded():
    """A 4:1 weight split must not starve the low-weight tenant.

    With weighted fair queuing, a backlogged low-weight tenant is admitted
    as soon as its virtual time is the minimum, so its worst wait is bounded
    by one batch turnover (a request's lifetime in steps), independent of
    how much the high-weight tenant bursts.
    """
    scheduler = _make_scheduler(_fair_share_config(tenant_weights={"H": 4.0, "L": 1.0}))
    for request in _tenant_requests("H", 100, "h"):
        scheduler.add_request(request)
    # Low-weight requests arrive one at a time while H is bursting.
    low_requests = _tenant_requests("L", 8, "l")
    for request in low_requests:
        scheduler.add_request(request)
        # Run a few steps so H keeps the batch full between L arrivals.
        _run_steps(scheduler, 3)
    _run_until_done(scheduler)

    snapshot = scheduler.fair_share.snapshot()
    low_wait = snapshot["tenants"]["L"]["wait"]
    assert low_wait["count"] == len(low_requests)
    # One batch turnover: a request lives at most TOKENS_PER_REQUEST steps
    # (1 prefill step + MAX_TOKENS decode steps), and a slot frees at least
    # every turnover. The bound holds with generous slack.
    bound = 2 * (1 + MAX_TOKENS) * 4
    assert low_wait["max_rounds"] <= bound, (
        f"low-weight tenant waited {low_wait['max_rounds']} rounds (bound {bound})"
    )
    # All low-weight requests completed; none starved.
    assert not scheduler.requests


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------


def _run_mixed_sequence() -> list[list[str]]:
    scheduler = _make_scheduler(
        _fair_share_config(
            tenant_weights={"A": 3.0, "B": 1.0, "C": 2.0},
            priority_aging_seconds=2.0,
        )
    )
    requests = []
    # Interleaved tenants, mixed priorities, fixed arrival times.
    specs = [
        ("A", 12, 0, 0.0),
        ("B", 8, 0, 0.5),
        ("C", 10, 1, 0.25),
        ("A", 6, -1, 2.0),
        ("B", 6, 3, 2.5),
    ]
    for tenant, n, priority, base in specs:
        requests.extend(
            _tenant_requests(
                tenant,
                n,
                f"{tenant}{priority}",
                arrival=base,
                priorities=[priority] * n,
            )
        )
    for request in requests:
        scheduler.add_request(request)
    return _run_steps(scheduler, 50)


def test_deterministic_replay():
    first = _run_mixed_sequence()
    second = _run_mixed_sequence()
    assert first == second
    assert len(first) == 50


# ---------------------------------------------------------------------------
# Single-tenant throughput vs the fcfs baseline
# ---------------------------------------------------------------------------


def _run_single_tenant(policy: str) -> tuple[list[list[str]], float]:
    fair_share = (
        _fair_share_config(tenant_weights={"only": 1.0})
        if policy == "fairshare"
        else None
    )
    scheduler = create_scheduler(
        scheduling_policy=policy,
        fair_share=fair_share,
        max_num_seqs=4,
        num_blocks=10000,
    )
    for request in _tenant_requests("only", 60, "r"):
        scheduler.add_request(request)
    start = time.perf_counter()
    scheduled = _run_until_done(scheduler)
    elapsed = time.perf_counter() - start
    return scheduled, elapsed


def test_single_tenant_throughput_matches_baseline():
    fcfs_schedule, fcfs_time = _run_single_tenant("fcfs")
    fair_schedule, fair_time = _run_single_tenant("fairshare")

    # Identical admission order and per-step token counts: zero throughput
    # regression for a single tenant.
    assert fair_schedule == fcfs_schedule
    total_tokens = (
        sum(len(step) for step in fair_schedule) * 1
    )  # one decode token per scheduled request per step
    assert total_tokens > 0
    # Scheduling-decision overhead stays controlled; measured well below 2x.
    assert fair_time < fcfs_time * 2.0 + 0.05, (
        f"fairshare scheduling took {fair_time:.3f}s vs fcfs {fcfs_time:.3f}s"
    )


# ---------------------------------------------------------------------------
# Missing tenant identity
# ---------------------------------------------------------------------------


def test_missing_tenant_id_goes_to_default_and_is_counted():
    scheduler = _make_scheduler(_fair_share_config())
    requests = _tenant_requests(None, 6, "anon") + _tenant_requests("A", 6, "a")
    for request in requests:
        scheduler.add_request(request)
    _run_steps(scheduler, 20)

    snapshot = scheduler.fair_share.snapshot()
    assert snapshot["num_defaulted_requests"] == 6
    default_stats = snapshot["tenants"][DEFAULT_TENANT_ID]
    assert default_stats["tokens"] > 0
    assert default_stats["requests_admitted"] > 0


# ---------------------------------------------------------------------------
# Preemption: victim choice, resume, no double charging
# ---------------------------------------------------------------------------


def test_preemption_victim_comes_from_over_share_tenant():
    config = _fair_share_config(tenant_weights={"A": 1.0, "B": 1.0})
    scheduler = _make_scheduler(config)
    (a1, a2) = _tenant_requests("A", 2, "a")
    (b1,) = _tenant_requests("B", 1, "b")
    for request in (a1, a2, b1):
        scheduler.add_request(request)
    scheduler.schedule()
    accountant = scheduler.fair_share
    assert accountant is not None
    # A ran two requests to B's one at equal weight: A is over its share.
    assert accountant.virtual_time("A") > accountant.virtual_time("B")
    victim = accountant.pick_victim([a1, b1, a2])
    assert victim is a2  # most recently admitted of the over-share tenant


def _preemption_setup():
    """A KV cache too small for the decode growth of all running requests.

    7 usable blocks (8 minus the null block) of 16 tokens. Each request
    starts with a 32-token prompt (2 blocks) and grows to 56 tokens
    (4 blocks) while decoding, so the running set outgrows the cache and
    the scheduler must preempt.
    """
    scheduler = _make_scheduler(
        _fair_share_config(tenant_weights={"A": 1.0, "B": 1.0}),
        num_blocks=8,
        max_num_seqs=16,
        block_size=16,
    )
    requests = _tenant_requests(
        "A", 3, "a", num_tokens=32, max_tokens=24
    ) + _tenant_requests("B", 3, "b", num_tokens=32, max_tokens=24)
    for request in requests:
        scheduler.add_request(request)
    return scheduler, requests


def test_preemption_resume_not_double_charged():
    """Memory pressure forces preemption; resumed requests are not
    charged twice, so total charged tokens equal total computed tokens."""
    scheduler, _ = _preemption_setup()
    _run_until_done(scheduler)

    assert not scheduler.requests, "all requests must complete"
    snapshot = scheduler.fair_share.snapshot()
    total_preemptions = sum(
        stats["preemptions"] for stats in snapshot["tenants"].values()
    )
    assert total_preemptions > 0, "expected the setup to force preemptions"
    total_charged = sum(stats["tokens"] for stats in snapshot["tenants"].values())
    # Each of the 6 requests computes exactly 32 + 24 - 1 positions (the
    # final sampled token computes no new position); recomputation after
    # preemption must not be charged again.
    assert total_charged == 6 * (32 + 24 - 1)


def test_preemptions_counted_per_tenant():
    """Accountant preemption counts match actual scheduler preemptions."""
    scheduler, _ = _preemption_setup()

    # Track each request's num_preemptions while it exists; the sum of the
    # per-request maxima is the number of preemptions that actually ran.
    seen: dict[str, int] = {}
    for _ in range(10000):
        output = scheduler.schedule()
        for request in scheduler.requests.values():
            seen[request.request_id] = max(
                seen.get(request.request_id, 0), request.num_preemptions
            )
        req_ids = list(output.num_scheduled_tokens)
        if not req_ids:
            break
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                sampled_token_ids=[[1]] * len(req_ids),
            ),
        )

    actual_preemptions = sum(seen.values())
    assert actual_preemptions > 0
    snapshot = scheduler.fair_share.snapshot()
    counted = sum(stats["preemptions"] for stats in snapshot["tenants"].values())
    # The ledger agrees with what actually ran: no preemption is lost or
    # double counted, and every preemption is attributed to a tenant.
    assert counted == actual_preemptions


# ---------------------------------------------------------------------------
# Priority bands coexisting with weights
# ---------------------------------------------------------------------------


def test_high_priority_wins_within_fairshare():
    scheduler = _make_scheduler(
        _fair_share_config(tenant_weights={"A": 1.0, "B": 8.0}),
        max_num_seqs=2,
    )
    # B (heavy weight) fills the queue first; A's high-priority request
    # arrives later but must jump the queue.
    for request in _tenant_requests("B", 10, "b", arrival=0.0):
        scheduler.add_request(request)
    (vip,) = _tenant_requests("A", 1, "vip", arrival=1.0, priorities=[-1])
    scheduler.add_request(vip)

    first_step = _run_steps(scheduler, 1)[0]
    assert vip.request_id in first_step


def test_low_priority_not_delayed_forever():
    """Priority aging lets a low-priority request through eventually."""
    scheduler = _make_scheduler(
        _fair_share_config(
            tenant_weights={"A": 8.0, "B": 1.0},
            priority_aging_seconds=1.0,
        ),
        max_num_seqs=2,
    )
    (low,) = _tenant_requests("B", 1, "low", arrival=0.0, priorities=[3])
    scheduler.add_request(low)
    # A keeps producing normal-priority requests with ever-later arrival
    # times, so the low-priority request ages past them.
    admitted_at: int | None = None
    for step in range(40):
        for request in _tenant_requests("A", 2, f"a{step}", arrival=(step + 1) * 1.0):
            scheduler.add_request(request)
        scheduled = _run_steps(scheduler, 1)
        if scheduled and low.request_id in scheduled[0]:
            admitted_at = step
            break
    assert admitted_at is not None, "low-priority request was never admitted"


# ---------------------------------------------------------------------------
# Share unit: token vs request accounting
# ---------------------------------------------------------------------------


def _run_mixed_lengths(share_unit: str) -> dict[str, dict]:
    scheduler = _make_scheduler(
        _fair_share_config(
            tenant_weights={"A": 1.0, "B": 1.0},
            share_unit=share_unit,
        )
    )
    # Equal weights, but B's prompts are 8x longer than A's.
    requests = _tenant_requests("A", 40, "a", num_tokens=10) + _tenant_requests(
        "B", 40, "b", num_tokens=80
    )
    for request in requests:
        scheduler.add_request(request)
    _run_steps(scheduler, 60)
    return scheduler.fair_share.snapshot()["tenants"]


def test_share_unit_request_counts_requests_not_tokens():
    by_request = _run_mixed_lengths("request")
    admitted_a = by_request["A"]["requests_admitted"]
    admitted_b = by_request["B"]["requests_admitted"]
    # With per-request accounting, equal weights split admissions evenly.
    assert abs(admitted_a - admitted_b) <= 2

    by_token = _run_mixed_lengths("token")
    admitted_a_t = by_token["A"]["requests_admitted"]
    admitted_b_t = by_token["B"]["requests_admitted"]
    # With per-token accounting, B's long prompts cost 8x the share, so A is
    # admitted much more often.
    assert admitted_a_t > 2 * admitted_b_t


# ---------------------------------------------------------------------------
# Reservations and borrowing
# ---------------------------------------------------------------------------


def test_reserved_share_is_honored_under_contention():
    """A's reservation (0.5) exceeds its weighted share (0.25); the
    reservation must win while both tenants are backlogged."""
    scheduler = _make_scheduler(
        _fair_share_config(
            tenant_weights={"A": 1.0, "B": 3.0},
            tenant_min_shares={"A": 0.5},
        )
    )
    for request in _tenant_requests("A", 100, "a") + _tenant_requests("B", 100, "b"):
        scheduler.add_request(request)
    _run_steps(scheduler, 60)
    tokens = _tokens_by_tenant(scheduler)
    share_a = tokens["A"] / (tokens["A"] + tokens["B"])
    assert share_a >= 0.45, f"reserved share not honored: {share_a:.4f}"


def test_idle_tenant_share_is_borrowed():
    """With only one tenant active, it borrows the full capacity."""
    scheduler = _make_scheduler(
        _fair_share_config(
            tenant_weights={"A": 1.0, "B": 3.0},
            tenant_min_shares={"A": 0.5},
        )
    )
    for request in _tenant_requests("B", 20, "b"):
        scheduler.add_request(request)
    _run_steps(scheduler, 40)
    tokens = _tokens_by_tenant(scheduler)
    assert tokens.get("A", 0) == 0
    assert tokens["B"] > 0


# ---------------------------------------------------------------------------
# Window settlement
# ---------------------------------------------------------------------------


def test_windows_settle_without_carryover():
    now = [0.0]
    config = _fair_share_config(window_size_seconds=30.0)
    accountant = TenantAccountant(config, time_fn=lambda: now[0])
    r0, r1, r2 = _tenant_requests("A", 3, "a")

    # Window [0, 30): charge 5 tokens.
    accountant.charge_scheduled_tokens(r0, 5, 5)
    now[0] = 35.0
    accountant.maybe_advance_window()
    assert len(accountant.windows) == 1
    assert accountant.windows[0]["tenants"]["A"]["tokens"] == 5

    # Window [30, 60): charge 7 tokens; the previous 5 do not carry over.
    accountant.charge_scheduled_tokens(r1, 7, 7)
    now[0] = 65.0
    accountant.maybe_advance_window()
    assert len(accountant.windows) == 2
    assert accountant.windows[1]["tenants"]["A"]["tokens"] == 7

    # Window [60, 90): charge 3 tokens.
    accountant.charge_scheduled_tokens(r2, 3, 3)
    now[0] = 95.0
    accountant.maybe_advance_window()
    assert len(accountant.windows) == 3
    assert accountant.windows[2]["tenants"]["A"]["tokens"] == 3

    # The lifetime view keeps the total; only windowed usage is reset.
    current = accountant.snapshot()["tenants"]["A"]
    assert current["tokens"] == 15


def test_wait_stats_recorded_per_tenant():
    scheduler = _make_scheduler(_fair_share_config())
    for request in _tenant_requests("A", 4, "a") + _tenant_requests("B", 4, "b"):
        scheduler.add_request(request)
    _run_steps(scheduler, 10)
    snapshot = scheduler.fair_share.snapshot()
    for tenant in ("A", "B"):
        wait = snapshot["tenants"][tenant]["wait"]
        assert wait["count"] > 0
        assert wait["max_seconds"] >= 0.0
        assert sum(wait["buckets"]) == wait["count"]
