# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for prefix-cache-hit-first fair admission.

Covers ordering, the deferral deadline/release bound, probe budget and
degradation, config/policy gating, ticket lifecycle and stats windows.
"""

import pytest

from vllm.config import SchedulerConfig
from vllm.v1.core.sched.scheduler import AdmissionTicket
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler


def _model_output(req_ids: list[str]) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=req_ids,
        sampled_token_ids=[[0] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _setup_fair(num_tokens: int = 48, warm_count: int = 1, **kwargs):
    """A scheduler where `warm*` prompts are cached and `cold` is not.

    Returns `(scheduler, cold, warm_requests)`; none of cold/warm are added
    yet. A `seed` request sharing the warm prompt is driven once to populate
    the prefix cache.
    """
    kwargs.setdefault("prefix_fair_admission_window", 4)
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=16, **kwargs)
    warm_ids = ["warm"] if warm_count == 1 else [f"warm{i}" for i in range(warm_count)]
    seeded = create_requests(
        num_requests=warm_count + 1,
        num_tokens=num_tokens,
        same_prompt=True,
        req_ids=["seed"] + warm_ids,
    )
    seed = seeded[0]
    warm_requests = seeded[1:]
    # Index 1 of a distinct-prompt batch is a prompt nobody else shares; index
    # 0 would collide with the all-zero warm prompt.
    cold = create_requests(
        num_requests=2, num_tokens=num_tokens, req_ids=["unused", "cold"]
    )[1]

    scheduler.add_request(seed)
    output = scheduler.schedule()
    assert "seed" in output.num_scheduled_tokens
    scheduler.update_from_output(output, _model_output(["seed"]))
    scheduler.finish_requests("seed", RequestStatus.FINISHED_ABORTED)

    for warm in warm_requests:
        assert scheduler.kv_cache_manager.get_num_cached_tokens(warm) > 0
    assert scheduler.kv_cache_manager.get_num_cached_tokens(cold) == 0
    return scheduler, cold, warm_requests


def test_prefix_fair_admission_prefers_cached_prefix():
    scheduler, cold, (warm,) = _setup_fair(
        prefix_fair_admission_window=4,
        prefix_fair_admission_deadline_steps=8,
    )
    scheduler.add_request(cold)
    scheduler.add_request(warm)

    output = scheduler.schedule()
    assert [r.req_id for r in output.scheduled_new_reqs] == ["warm", "cold"]
    assert output.num_scheduled_tokens["warm"] < 48


def test_prefix_fair_admission_releases_after_deadline():
    scheduler, cold, (warm,) = _setup_fair(
        prefix_fair_admission_window=4,
        prefix_fair_admission_deadline_steps=1,
    )
    # Cold arrives first, so release-to-front leaves the queue unchanged and
    # exercises the "even with no actual exchange" accounting.
    scheduler.add_request(cold)
    scheduler.add_request(warm)

    for _ in range(3):
        scheduler._reorder_waiting_by_fair_cached_prefix()
        assert scheduler.waiting.peek_request().request_id == "cold"
        ticket = scheduler._admission_tickets["cold"]
        assert ticket.deferred_steps == 0

    stats = scheduler.make_stats().prefix_fair_admission_stats
    assert stats is not None
    assert stats.fairness_releases == 3
    assert stats.max_deferred_steps <= 1


def test_prefix_fair_admission_deferral_bound():
    scheduler, cold, warm_requests = _setup_fair(
        warm_count=3,
        prefix_fair_admission_window=4,
        prefix_fair_admission_deadline_steps=2,
    )
    scheduler.add_request(cold)
    for warm in warm_requests:
        scheduler.add_request(warm)

    for _ in range(10):
        scheduler._reorder_waiting_by_fair_cached_prefix()
        deferred = scheduler._admission_tickets["cold"].deferred_steps
        # Debt is consumed by a release as soon as it reaches the deadline.
        assert deferred <= 2

    stats = scheduler.make_stats().prefix_fair_admission_stats
    assert stats is not None
    assert stats.max_deferred_steps == 2


def test_prefix_fair_admission_probe_budget():
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        prefix_fair_admission_window=6,
        prefix_fair_admission_probe_budget=2,
    )
    requests = create_requests(
        num_requests=6, num_tokens=16, req_ids=[f"c{i}" for i in range(6)]
    )
    for request in requests:
        scheduler.add_request(request)

    scheduler._reorder_waiting_by_fair_cached_prefix()
    stats = scheduler.make_stats().prefix_fair_admission_stats
    assert stats is not None
    assert stats.probes == 2
    assert stats.probe_budget_exhausted == 4
    order_once = [r.request_id for r in scheduler.waiting]
    assert order_once == [f"c{i}" for i in range(6)]

    # Same input state (all-zero scores, arrival tie-break) gives the same
    # result on a second arbitration.
    scheduler._reorder_waiting_by_fair_cached_prefix()
    order_twice = [r.request_id for r in scheduler.waiting]
    assert order_twice == order_once


def test_prefix_fair_admission_zero_budget_keeps_arrival_order():
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        prefix_fair_admission_window=4,
        prefix_fair_admission_probe_budget=0,
        prefix_fair_admission_deadline_steps=100,
    )
    requests = create_requests(
        num_requests=4, num_tokens=16, req_ids=["a0", "a1", "a2", "a3"]
    )
    for request in requests:
        scheduler.add_request(request)

    scheduler._reorder_waiting_by_fair_cached_prefix()
    stats = scheduler.make_stats().prefix_fair_admission_stats
    assert stats is not None
    assert stats.probes == 0
    # Every one of the four candidates is unprobed/budget-exhausted.
    assert stats.probe_budget_exhausted == 4
    assert [r.request_id for r in scheduler.waiting] == [
        "a0",
        "a1",
        "a2",
        "a3",
    ]


def test_prefix_fair_admission_probe_error_degrades(monkeypatch: pytest.MonkeyPatch):
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        prefix_fair_admission_window=4,
        prefix_fair_admission_probe_budget=1,
    )
    requests = create_requests(num_requests=2, num_tokens=16, req_ids=["cold", "peer"])
    for request in requests:
        scheduler.add_request(request)

    def raise_runtime_error(_request):
        raise RuntimeError("probe boom")

    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "get_num_cached_tokens",
        raise_runtime_error,
    )
    # Must not propagate: schedule()-adjacent arbitration degrades in place.
    scheduler._reorder_waiting_by_fair_cached_prefix()
    assert scheduler._fair_admission_stats.probe_errors == 1
    assert [r.request_id for r in scheduler.waiting] == ["cold", "peer"]

    monkeypatch.undo()
    scheduler._reorder_waiting_by_fair_cached_prefix()
    stats = scheduler._fair_admission_stats
    # Counters keep accumulating after the replacement is withdrawn; the
    # earlier error is retained and the restored probe is counted.
    assert stats.probe_errors == 1
    assert stats.probes == 2


def test_prefix_fair_admission_exclusive_with_cache_aware():
    with pytest.raises(ValueError) as exc_info:
        SchedulerConfig.default_factory(
            cache_aware_admission_window=4,
            prefix_fair_admission_window=4,
        )
    message = str(exc_info.value)
    assert "prefix_fair_admission_window" in message
    assert "cache_aware_admission_window" in message


def test_prefix_fair_admission_inactive_for_priority_policy():
    scheduler = create_scheduler(
        scheduling_policy="priority",
        enable_prefix_caching=True,
        prefix_fair_admission_window=4,
    )
    assert scheduler.fair_admission_window == 0
    assert scheduler.fair_admission_inactive_reason == "priority_policy"

    requests = create_requests(
        num_requests=3, num_tokens=16, req_ids=["low", "high", "mid"]
    )
    requests[0].priority = 5
    requests[1].priority = 0
    requests[2].priority = 2
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert [r.req_id for r in output.scheduled_new_reqs] == [
        "high",
        "mid",
        "low",
    ]


def test_prefix_fair_admission_inactive_without_prefix_caching():
    scheduler = create_scheduler(
        enable_prefix_caching=False,
        prefix_fair_admission_window=4,
    )
    assert scheduler.fair_admission_window == 0
    assert scheduler.fair_admission_inactive_reason == "prefix_caching_disabled"
    assert scheduler._fair_admission_stats.probes == 0


def test_prefix_fair_admission_tickets_released():
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        prefix_fair_admission_window=4,
    )
    requests = create_requests(
        num_requests=4, num_tokens=16, req_ids=["r0", "r1", "r2", "r3"]
    )
    for request in requests:
        scheduler.add_request(request)

    assert set(scheduler._admission_tickets) == set(scheduler.requests)
    assert all(
        isinstance(ticket, AdmissionTicket)
        for ticket in scheduler._admission_tickets.values()
    )

    # Ordinary finish path deletes the ticket immediately.
    scheduler.finish_requests("r0", RequestStatus.FINISHED_ABORTED)
    assert "r0" not in scheduler._admission_tickets

    # Delayed block-free path (async KV abort): _free_request runs with
    # delay_free_blocks=True, so self.requests may retain the entry but the
    # ticket must already be gone.
    delayed = requests[1]
    delayed.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.finish_requests("r1", RequestStatus.FINISHED_ABORTED)
    assert "r1" not in scheduler._admission_tickets
    assert "r1" in scheduler.requests

    scheduler.finish_requests(["r2", "r3"], RequestStatus.FINISHED_ABORTED)
    assert scheduler._admission_tickets == {}
    assert set(scheduler._admission_tickets) <= set(scheduler.requests)


def test_prefix_fair_admission_reset_prefix_cache_keeps_debt():
    scheduler, cold, (warm,) = _setup_fair(
        prefix_fair_admission_window=4,
        prefix_fair_admission_deadline_steps=2,
    )
    scheduler.add_request(cold)
    scheduler.add_request(warm)
    scheduler._reorder_waiting_by_fair_cached_prefix()

    ticket = scheduler._admission_tickets["cold"]
    assert ticket.deferred_steps == 1
    assert ticket.probe_valid is True
    assert ticket.last_probe_tokens == 0  # cold is a miss
    enqueue_step = ticket.enqueue_step

    assert scheduler.reset_prefix_cache() is True

    assert ticket.probe_valid is False
    assert ticket.last_probe_tokens == 0
    assert ticket.deferred_steps == 1
    assert ticket.enqueue_step == enqueue_step


def test_prefix_fair_admission_stats_window_reset():
    scheduler, cold, (warm,) = _setup_fair(
        prefix_fair_admission_window=4,
        prefix_fair_admission_deadline_steps=8,
    )
    scheduler.add_request(cold)
    scheduler.add_request(warm)
    scheduler._reorder_waiting_by_fair_cached_prefix()

    first = scheduler.make_stats()
    assert first is not None
    fair_first = first.prefix_fair_admission_stats
    assert fair_first is not None
    assert fair_first.steps > 0
    assert first.prefix_cache_stats is not None

    second = scheduler.make_stats()
    assert second is not None
    fair_second = second.prefix_fair_admission_stats
    assert fair_second is not None
    assert fair_second.steps == 0
    assert fair_second.fairness_releases == 0
    # Prefix cache stats remain an independent, populated structure.
    assert second.prefix_cache_stats is not None


def test_prefix_fair_admission_empty_and_short_queue_no_stats():
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        prefix_fair_admission_window=4,
    )
    # Empty queue must return without touching peek_request.
    scheduler._reorder_waiting_by_fair_cached_prefix()
    assert scheduler._fair_admission_stats.steps == 0

    (one,) = create_requests(num_requests=1, num_tokens=16, req_ids=["solo"])
    scheduler.add_request(one)
    scheduler._reorder_waiting_by_fair_cached_prefix()
    assert scheduler._fair_admission_stats.steps == 0


def test_prefix_fair_admission_disabled_reason():
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        prefix_fair_admission_window=0,
    )
    assert scheduler.fair_admission_window == 0
    assert scheduler.fair_admission_inactive_reason == "disabled"


def test_prefix_fair_admission_keeps_fixed_slots_in_place():
    scheduler, cold, (warm,) = _setup_fair(
        prefix_fair_admission_window=5,
        prefix_fair_admission_deadline_steps=100,
    )
    blocked, started, tail = create_requests(
        num_requests=3, num_tokens=48, req_ids=["blocked", "started", "tail"]
    )
    # cold, blocked, started, warm, tail
    for request in (cold, blocked, started, warm, tail):
        scheduler.add_request(request)
    blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    started.num_computed_tokens = 16

    scheduler._reorder_waiting_by_fair_cached_prefix()
    after = [r.request_id for r in scheduler.waiting]
    # Only cold and warm are candidates; warm wins and swaps into slot 0 while
    # blocked/started fixed indices and the out-of-window tail stay put.
    assert after == ["warm", "blocked", "started", "cold", "tail"]


def test_prefix_fair_admission_preemption_invalidates_probe_keeps_debt():
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        prefix_fair_admission_window=4,
        prefix_fair_admission_deadline_steps=2,
        num_blocks=10000,
    )
    (request,) = create_requests(num_requests=1, num_tokens=16, req_ids=["r0"])
    scheduler.add_request(request)
    # Drive it to RUNNING so _preempt_request's assert holds.
    output = scheduler.schedule()
    assert "r0" in output.num_scheduled_tokens
    scheduler.update_from_output(output, _model_output(["r0"]))
    assert request.status != RequestStatus.PREEMPTED

    ticket = scheduler._admission_tickets["r0"]
    ticket.deferred_steps = 1
    ticket.probe_valid = True
    ticket.last_probe_tokens = 32

    scheduler._preempt_request(request, timestamp=0.0)

    assert request.status == RequestStatus.PREEMPTED
    assert scheduler.waiting.peek_request().request_id == "r0"
    assert ticket.probe_valid is False
    assert ticket.last_probe_tokens == 0
    assert ticket.deferred_steps == 1
