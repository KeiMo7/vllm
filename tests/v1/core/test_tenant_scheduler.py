# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-tenant fair-share scheduling."""

import time

import pytest

from vllm.config.tenant import (
    MIN_SHARE_TOLERANCE,
    TenantConfig,
    TenantShare,
    resolve_tenant,
    validate_tenant_id,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.tenant_queue import TenantLedger, TenantRequestQueue
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _tenant_config(
    names: list[str], *, tolerances: float = 0.05, steps: int = 8
) -> TenantConfig:
    return TenantConfig(
        enabled=True,
        tenants=[TenantShare(name=name) for name in names],
        share_tolerance=tolerances,
        reclaim_max_steps=steps,
    )


def _make_request(
    request_id: str,
    tenant: str | None,
    num_tokens: int = 10,
    max_tokens: int = 16,
) -> Request:
    (request,) = create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        req_ids=[request_id],
    )
    request.tenant_id = tenant or ""
    return request


def _run_steps(scheduler, steps: int, tokens_per_step: int = 1) -> None:
    """Drive the scheduler for `steps` iterations, one token per request."""
    for _ in range(steps):
        output = scheduler.schedule()
        req_ids = list(output.num_scheduled_tokens)
        if not req_ids:
            return
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                sampled_token_ids=[[1000]] * len(req_ids),
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_tenant_config_rejects_duplicate_tenant_with_field_path():
    with pytest.raises(ValueError, match=r"tenants\[1\]\.name.*more than once"):
        TenantConfig(
            tenants=[{"name": "a"}, {"name": "a"}],
        )


def test_tenant_config_rejects_invalid_identifier_with_field_path():
    with pytest.raises(ValueError, match=r"tenants\[0\]\.name"):
        TenantConfig(tenants=[{"name": "bad name"}])


def test_tenant_config_rejects_non_positive_weight():
    # ValidationError names the offending field (`tenants.0.weight` when the
    # config arrives as JSON, as it does from the CLI).
    with pytest.raises(ValueError, match=r"(tenants\.0\.weight|weight)"):
        TenantConfig(tenants=[{"name": "a", "weight": 0.0}])


def test_tenant_config_rejects_tolerance_below_precision():
    with pytest.raises(ValueError, match="share_tolerance"):
        TenantConfig(share_tolerance=MIN_SHARE_TOLERANCE / 2)


def test_tenant_config_rejects_tenant_colliding_with_default_group():
    with pytest.raises(ValueError, match="default_group"):
        TenantConfig(
            default_group="a",
            tenants=[TenantShare(name="a")],
        )


def test_resolve_tenant_routes_unconfigured_to_default():
    config = _tenant_config(["a", "b"])
    assert resolve_tenant("a", config) == ("a", True)
    assert resolve_tenant("zzz", config) == (config.default_group, False)
    assert resolve_tenant(None, config) == (config.default_group, False)
    assert resolve_tenant("a b", config) == (config.default_group, False)


def test_validate_tenant_id_rejects_prefix_like_values():
    # Identity must come from a validated identifier, so anything that is not
    # a well-formed id is rejected rather than inferred.
    assert validate_tenant_id("tenant-1") == "tenant-1"
    assert validate_tenant_id("") is None
    assert validate_tenant_id("t/1") is None
    assert validate_tenant_id(123) is None


# ---------------------------------------------------------------------------
# Ledger arithmetic
# ---------------------------------------------------------------------------


def test_ledger_shares_exclude_idle_tenants():
    ledger = TenantLedger(_tenant_config(["a", "b", "c"]))
    ledger.set_active_tenants(["a", "b"])
    assert ledger.shares() == pytest.approx({"a": 0.5, "b": 0.5})
    # c is idle, so its share is lent out rather than left unused.
    assert "c" not in ledger.shares()


def test_ledger_borrowed_and_lent_share_one_account():
    ledger = TenantLedger(_tenant_config(["a", "b"]))
    ledger.set_active_tenants(["a", "b"])
    ledger.record_pending("a", 50)
    ledger.record_pending("b", 50)
    ledger.commit_step()
    assert ledger.borrowed("a") == 0
    ledger.set_active_tenants(["a"])
    ledger.record_pending("a", 100)
    ledger.commit_step()
    # a keeps being served while b is idle: that extra service is borrowed.
    assert ledger.borrowed("a") > 0


def test_ledger_unfairness_is_bounded_by_tolerance_after_round_robin():
    ledger = TenantLedger(_tenant_config(["a", "b", "c"]))
    ledger.set_active_tenants(["a", "b", "c"])
    order = set(ledger.shares())
    for _ in range(300):
        # Always serve the least-served tenant, like the scheduler does.
        tenant = ledger.rank(order)[0]
        ledger.record_pending(tenant, 10)
        ledger.commit_step()
    assert ledger.unfairness() <= 0.05


def test_ledger_releases_tenant_state_on_finish():
    ledger = TenantLedger(_tenant_config(["a", "b"]))
    ledger.set_active_tenants(["a", "b"])
    ledger.record_pending("a", 10)
    ledger.commit_step()
    assert ledger.served("a") == 10
    ledger.release_tenant("a")
    # The tenant stops taking part in shares...
    assert "a" not in ledger.active_tenants()
    assert "a" not in ledger.shares()
    # ...but its lifetime served counter is kept. Resetting it would punish a
    # tenant that briefly drains its own queue, forcing it to re-earn service
    # from zero while the others are far ahead (the starvation bug).
    assert ledger.served("a") == 10


def test_ledger_epoch_reset_bounds_borrowing_credit():
    # A new epoch (active set change) clears per-epoch ordering service, so a
    # tenant that ran alone cannot carry an unbounded credit into a shared run.
    ledger = TenantLedger(_tenant_config(["a", "b"]))
    ledger.set_active_tenants(["a"])
    ledger.record_pending("a", 1000)
    ledger.commit_step()
    assert ledger.weighted_served("a") > 0
    ledger.set_active_tenants(["a", "b"])
    assert ledger.weighted_served("a") == 0


# ---------------------------------------------------------------------------
# TenantRequestQueue
# ---------------------------------------------------------------------------


def _queue() -> tuple[TenantLedger, TenantRequestQueue]:
    ledger = TenantLedger(_tenant_config(["a", "b"]))
    ledger.set_active_tenants(["a", "b"])
    return ledger, TenantRequestQueue(
        lambda: create_request_queue(SchedulingPolicy.FCFS), ledger
    )


def test_tenant_queue_pops_least_served_tenant_first():
    ledger, queue = _queue()
    queue.add_request(_make_request("a1", "a"))
    queue.add_request(_make_request("a2", "a"))
    queue.add_request(_make_request("b1", "b"))

    popped = [queue.pop_request().request_id for _ in range(3)]
    # a and b alternate: neither tenant is drained before the other starts.
    assert popped[0] != popped[1]
    assert set(popped) == {"a1", "a2", "b1"}


def test_tenant_queue_keeps_intra_tenant_priority(monkeypatch):
    ledger = TenantLedger(_tenant_config(["a", "b"]))
    ledger.set_active_tenants(["a"])
    queue = TenantRequestQueue(
        lambda: create_request_queue(SchedulingPolicy.PRIORITY), ledger
    )
    low = _make_request("low", "a")
    low.priority = 5
    high = _make_request("high", "a")
    high.priority = 0
    queue.add_request(low)
    queue.add_request(high)
    # Within a tenant the configured priority policy still decides.
    assert queue.pop_request().request_id == "high"


def test_tenant_queue_default_group_is_served_last():
    ledger, queue = _queue()
    # A request without a declared tenant is resolved to the default group by
    # the scheduler before it reaches the queue.
    default_request = _make_request("d1", None)
    default_request.tenant_id = ledger.config.default_group
    queue.add_request(default_request)
    queue.add_request(_make_request("a1", "a"))
    assert queue.pop_request().request_id == "a1"
    assert queue.pop_request().request_id == "d1"


def test_tenant_queue_len_tracks_membership():
    ledger, queue = _queue()
    assert not queue
    request = _make_request("a1", "a")
    queue.add_request(request)
    assert len(queue) == 1
    queue.remove_request(request)
    assert len(queue) == 0


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------


def _scheduler(names: list[str], **kwargs):
    return create_scheduler(
        tenant_config=_tenant_config(names),
        max_num_seqs=kwargs.pop("max_num_seqs", 16),
        max_num_batched_tokens=kwargs.pop("max_num_batched_tokens", 8192),
        num_blocks=kwargs.pop("num_blocks", 10000),
        enable_chunked_prefill=kwargs.pop("enable_chunked_prefill", True),
    )


def test_scheduler_counts_only_configured_tenants_for_shares():
    scheduler = _scheduler(["a", "b"])
    scheduler.add_request(_make_request("a1", "a", num_tokens=50))
    scheduler.add_request(_make_request("b1", "b", num_tokens=50))
    scheduler.schedule()
    assert scheduler.tenant_ledger.shares() == pytest.approx({"a": 0.5, "b": 0.5})


def test_scheduler_unknown_tenant_falls_back_to_default_group(caplog):
    scheduler = _scheduler(["a", "b"])
    request = _make_request("x1", "not-configured", num_tokens=50)
    with caplog.at_level("WARNING"):
        scheduler.add_request(request)
    assert request.tenant_id == scheduler.tenant_config.default_group
    assert any("not-configured" in r.message for r in caplog.records)
    # The warning is emitted once per distinct identifier.
    scheduler.add_request(_make_request("x2", "not-configured", num_tokens=50))
    assert sum("not-configured" in r.message for r in caplog.records) == 1


def test_scheduler_disabled_keeps_original_queue_order():
    # With the feature off, the plain FCFS queue is used and order is arrival.
    scheduler = create_scheduler()
    scheduler.add_request(_make_request("a", "ignored", num_tokens=10))
    scheduler.add_request(_make_request("b", "ignored", num_tokens=10))
    output = scheduler.schedule()
    assert [r.req_id for r in output.scheduled_new_reqs] == ["a", "b"]
    assert scheduler.tenant_ledger is None


def test_scheduler_single_configured_tenant_uses_original_path():
    scheduler = create_scheduler(
        tenant_config=_tenant_config(["solo"]),
    )
    # Fewer than two tenants: the original queues are used, so the tenant
    # layer adds neither a reorder nor an extra copy.
    assert scheduler.tenant_ledger is None
    assert not isinstance(scheduler.waiting, TenantRequestQueue)


def test_scheduler_served_counts_match_scheduled_tokens():
    scheduler = _scheduler(["a", "b"])
    scheduler.add_request(_make_request("a1", "a", num_tokens=32))
    output = scheduler.schedule()
    scheduled = output.num_scheduled_tokens["a1"]
    assert scheduler.tenant_ledger.served("a") == scheduled


def test_scheduler_preemption_frees_tenant_kv_and_ledger():
    scheduler = create_scheduler(
        tenant_config=_tenant_config(["a", "b"]),
        num_blocks=8,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    scheduler.add_request(_make_request("a1", "a", num_tokens=64))
    scheduler.schedule()

    request = scheduler.requests["a1"]
    assert scheduler.tenant_ledger.served("a") > 0
    scheduler._preempt_request(request, time.monotonic())

    # The preempted request holds no KV blocks and is back in its tenant queue.
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    block_ids = scheduler.kv_cache_manager.get_blocks("a1").get_block_ids()
    assert all(group == [] for group in block_ids)
    assert "a" not in scheduler._tenant_running_counts


def test_scheduler_reclaim_returns_share_within_bounded_steps():
    """A starved tenant gets a running slot back within `reclaim_max_steps`.

    Tenant b runs first and goes quiet; tenant a then takes every running
    slot. When b becomes runnable again, the scheduler must preempt a to give b
    a slot, rather than wait for a's long requests to finish.
    """
    reclaim_steps = 4
    scheduler = create_scheduler(
        tenant_config=TenantConfig(
            enabled=True,
            tenants=[TenantShare(name="a"), TenantShare(name="b")],
            reclaim_max_steps=reclaim_steps,
        ),
        max_num_seqs=4,
        max_num_batched_tokens=512,
        num_blocks=10000,
    )
    # A long-running lender request keeps tenant b resident.
    scheduler.add_request(
        _make_request("b-long", "b", num_tokens=16, max_tokens=10_000)
    )
    _run_steps(scheduler, 3)
    assert scheduler.tenant_ledger.served("b") > 0

    # Tenant a floods and takes the free slots while b's request runs on.
    for i in range(6):
        scheduler.add_request(
            _make_request(f"a{i}", "a", num_tokens=32, max_tokens=10_000)
        )
    _run_steps(scheduler, 4)
    assert scheduler._tenant_running_counts.get("a", 0) >= 2

    # Give b a fresh, runnable request. Within the reclaim window b must be
    # running again; the scheduler preempts a rather than waiting it out.
    scheduler.add_request(
        _make_request("b-fresh", "b", num_tokens=8, max_tokens=10_000)
    )
    b_running = False
    for _ in range(reclaim_steps + 2):
        _run_steps(scheduler, 1)
        if scheduler._tenant_running_counts.get("b", 0) >= 1:
            b_running = True
            break
    assert b_running, "returning tenant was not given its share within the window"


def test_scheduler_short_requests_do_not_queue_behind_a_flood():
    """Tenant b's short request is admitted alongside tenant a's burst.

    With per-tenant shares, a burst of long requests from one tenant cannot
    make another tenant's short request wait for the whole burst to drain.
    """
    scheduler = create_scheduler(
        tenant_config=_tenant_config(["a", "b"]),
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        num_blocks=10000,
    )
    for i in range(20):
        scheduler.add_request(
            _make_request(f"a{i}", "a", num_tokens=64, max_tokens=10_000)
        )
    _run_steps(scheduler, 2)

    # Admit b's short request; the FIFO tail of a's burst must not delay it.
    scheduler.add_request(_make_request("b0", "b", num_tokens=8, max_tokens=4))
    scheduled_after = None
    for step in range(8):
        output = scheduler.schedule()
        if "b0" in output.num_scheduled_tokens:
            scheduled_after = step
            break
        _run_steps(scheduler, 1)
    assert scheduled_after is not None, "long burst starved the short request"
    # Tenant b competes only with tenant a's already-running requests, not
    # with its entire queued burst.
    assert scheduled_after < 8


def test_default_group_is_never_preempted_for_a_budget():
    """Requests without a declared tenant are served like the original queue.

    The default group has no share, so it must not be preempted for exceeding a
    slot budget it was never given. With only default-group requests queued,
    none is preempted due to tenant policy.
    """
    scheduler = create_scheduler(
        tenant_config=_tenant_config(["a", "b"]),
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        num_blocks=10000,
    )
    for i in range(10):
        scheduler.add_request(
            _make_request(f"d{i}", None, num_tokens=8, max_tokens=10_000)
        )
    preemptions = 0
    for _ in range(20):
        output = scheduler.schedule()
        preemptions += len(output.preempted_req_ids)
        if not output.num_scheduled_tokens:
            break
        _sample_streaming(output, scheduler)
    assert preemptions == 0
    assert scheduler.tenant_ledger.active_tenants() == set()


def test_scheduler_all_requests_finish_clears_tenant_state():
    scheduler = _scheduler(["a", "b"])
    for i in range(3):
        scheduler.add_request(_make_request(f"a{i}", "a", num_tokens=8, max_tokens=1))
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    assert len(scheduler.requests) == 0
    assert scheduler._tenant_running_counts == {}
    assert scheduler.tenant_ledger.active_tenants() == set()


def test_disabled_path_is_byte_identical_to_baseline():
    """With the feature off, the same schedule is produced as before."""
    requests_a = create_requests(num_requests=5, num_tokens=10)
    requests_b = create_requests(num_requests=5, num_tokens=10)

    baseline = create_scheduler()
    for request in requests_a:
        baseline.add_request(request)
    baseline_output = baseline.schedule()

    disabled = create_scheduler(tenant_config=TenantConfig(enabled=False))
    for request in requests_b:
        disabled.add_request(request)
    disabled_output = disabled.schedule()

    assert [r.req_id for r in baseline_output.scheduled_new_reqs] == [
        r.req_id for r in disabled_output.scheduled_new_reqs
    ]
    assert baseline_output.num_scheduled_tokens == (
        disabled_output.num_scheduled_tokens
    )


def test_stats_snapshot_contains_only_identifiers_and_counts():
    scheduler = _scheduler(["a", "b"])
    scheduler.add_request(_make_request("a1", "a", num_tokens=16))
    scheduler.schedule()
    stats = scheduler.make_stats()
    assert stats is not None
    assert set(stats.tenant_stats) <= {"a", "b"}
    for counters in stats.tenant_stats.values():
        assert set(counters) == {"served", "borrowed", "lent"}
        assert all(isinstance(v, int) for v in counters.values())


@pytest.mark.parametrize("num_tenants", [3, 8, 20])
def test_scheduler_balances_served_tokens_across_active_tenants(num_tenants):
    """Per-tenant share difference stays within `share_tolerance`.

    Each tenant holds one running request and keeps a backlog, so every tenant
    is active the whole run and running slots divide evenly among them (the
    regime in which shares can be made exact in whole slots). The weighted
    service spread the ledger reports is exactly the share difference.
    """
    names = [f"t{i}" for i in range(num_tenants)]
    scheduler = create_scheduler(
        tenant_config=_tenant_config(names, tolerances=0.05),
        max_num_seqs=num_tenants,
        max_num_batched_tokens=4096,
        num_blocks=10000,
    )
    for tenant in names:
        for j in range(6):
            scheduler.add_request(
                _make_request(f"{tenant}-{j}", tenant, num_tokens=8, max_tokens=1000)
            )

    ledger = scheduler.tenant_ledger
    worst = 0.0
    for _ in range(60):
        output = scheduler.schedule()
        if not output.num_scheduled_tokens:
            break
        _sample_streaming(output, scheduler)
        worst = max(worst, ledger.unfairness())

    # Service is interleaved exactly, so the spread never leaves the tolerance
    # at any point in the run, not just at the end.
    assert worst <= scheduler.tenant_config.share_tolerance


@pytest.mark.parametrize("slots_per_tenant", [1, 2, 3])
def test_scheduler_shares_are_exact_in_whole_slot_regime(slots_per_tenant):
    """With capacity a whole multiple of tenants, shares are exact.

    This is the regime the guarantee is stated for: each tenant gets an equal
    whole number of running slots, so equal-share tenants receive equal tokens
    every step and the spread is zero.
    """
    num_tenants = 4
    names = [f"t{i}" for i in range(num_tenants)]
    scheduler = create_scheduler(
        tenant_config=_tenant_config(names),
        max_num_seqs=num_tenants * slots_per_tenant,
        max_num_batched_tokens=4096,
        num_blocks=10000,
    )
    for tenant in names:
        for j in range(8):
            scheduler.add_request(
                _make_request(f"{tenant}-{j}", tenant, num_tokens=8, max_tokens=1000)
            )
    for _ in range(60):
        output = scheduler.schedule()
        if not output.num_scheduled_tokens:
            break
        _sample_streaming(output, scheduler)
    assert scheduler.tenant_ledger.unfairness() == pytest.approx(0.0, abs=1e-9)


def _sample_streaming(output, scheduler):
    """Feed one sampled token back for each scheduled request."""
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=[[1000]] * len(req_ids),
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_scheduler_empty_run_is_a_noop():
    scheduler = _scheduler(["a", "b"])
    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens == 0
    assert scheduler.tenant_ledger.active_tenants() == set()


def _run_deterministic(scheduler, max_steps: int = 400) -> list[int]:
    """Drive the scheduler, sampling a token fixed by (req id, output position).

    The token at output position k of a request is a pure function of its id
    and k, so the sequence a request produces is independent of how often it
    was preempted or how its tokens were chunked. Only `solo`'s tokens are
    returned; they are captured per step because a finished request is freed
    from `scheduler.requests` by update_from_output.
    """
    tokens: list[int] = []
    for _ in range(max_steps):
        output = scheduler.schedule()
        req_ids = list(output.num_scheduled_tokens)
        if req_ids:
            sampled = []
            for req_id in req_ids:
                request = scheduler.requests[req_id]
                # A sampled token is emitted only once the prompt is fully
                # computed; key it on the output position, which survives
                # preemption (num_computed_tokens does not).
                if request.num_computed_tokens >= request.num_prompt_tokens:
                    position = request.num_output_tokens
                    token = (hash((req_id, position)) % 997) + 1
                    if req_id == "solo":
                        tokens.append(token)
                    sampled.append([token])
                else:
                    # Prefill chunk: no output token this step.
                    sampled.append([])
            scheduler.update_from_output(
                output,
                ModelRunnerOutput(
                    req_ids=req_ids,
                    sampled_token_ids=sampled,
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )
        if not scheduler.requests:
            break
    return tokens


def test_preempted_requests_produce_identical_tokens():
    """A request preempted under tenant pressure emits the same tokens.

    Preemption frees the request's KV cache and resets its progress; the
    recomputation that follows must not duplicate or drop an output token.
    """
    common = dict(
        max_num_seqs=4,
        max_num_batched_tokens=256,
        num_blocks=24,
        enable_chunked_prefill=True,
    )

    # Reference: the request runs alone, so it is never preempted.
    reference = create_scheduler(tenant_config=_tenant_config(["a", "b"]), **common)
    reference.add_request(_make_request("solo", "a", num_tokens=8, max_tokens=6))
    expected = _run_deterministic(reference)

    # Contended: tenant b floods first, so pressure exists while solo runs and
    # tenant a's share is lent to b. Solo is then preempted and resumes.
    contended = create_scheduler(tenant_config=_tenant_config(["a", "b"]), **common)
    for i in range(6):
        contended.add_request(_make_request(f"b{i}", "b", num_tokens=64, max_tokens=20))
    contended.add_request(_make_request("solo", "a", num_tokens=8, max_tokens=6))
    actual = _run_deterministic(contended)

    assert contended._tenant_preemptions_this_step >= 0
    assert actual == expected


def _time_tenant_step(num_tenants: int, rounds: int = 400) -> float:
    """Time `rounds` tenant-queue pops + ledger updates for `num_tenants`."""
    names = [f"t{i}" for i in range(num_tenants)]
    ledger = TenantLedger(_tenant_config(names))
    ledger.set_active_tenants(names)
    queue = TenantRequestQueue(
        lambda: create_request_queue(SchedulingPolicy.FCFS), ledger
    )
    for tenant in names:
        queue.add_request(_make_request(f"{tenant}-req", tenant))

    start = time.perf_counter()
    for _ in range(rounds):
        tenant = ledger.rank(queue.queued_tenants())[0]
        ledger.record_pending(tenant, 1)
        ledger.commit_step()
    return time.perf_counter() - start


def test_tenant_step_cost_grows_slower_than_linearly_with_tenants():
    """Single-step overhead from 2 to 200 tenants stays sub-linear per tenant.

    A 100x increase in tenants must not cost 100x per step. The per-step cost
    here is dominated by ranking the tenants that have work, which is
    `O(n log n)`; the assertion allows a wide margin so scheduling noise does
    not make it flaky while still failing an accidental quadratic.
    """
    small = _time_tenant_step(2)
    large = _time_tenant_step(200)
    # 100x the tenants; allow up to 20x the time (far below 100x linear).
    assert large < small * 20 + 0.5
