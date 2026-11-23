# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate the fair-share acceptance measurements into reports/.

Run from the repository root:

    .venv/bin/python reports/generate_measurements.py

Reproduces the numbers asserted in
tests/v1/core/test_fair_share_scheduler.py:

1. share deviation under a two-tenant burst;
2. worst waiting time of the low-weight tenant vs. its bound;
3. replay determinism of the schedule;
4. single-tenant throughput and scheduling overhead vs. the fcfs baseline.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.v1.core.utils import create_requests, create_scheduler  # noqa: E402
from vllm.config.scheduler import FairShareConfig  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent


def run(scheduler, num_steps):
    scheduled = []
    for _ in range(num_steps):
        output = scheduler.schedule()
        req_ids = list(output.num_scheduled_tokens)
        if not req_ids:
            break
        scheduled.append(req_ids)
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=[[1]] * len(req_ids)),
        )
    return scheduled


def make_requests(tenant, n, prefix, arrival=0.0, **kwargs):
    kwargs.setdefault("num_tokens", 10)
    kwargs.setdefault("max_tokens", 5)
    return create_requests(
        n,
        req_ids=[f"{prefix}-{i}" for i in range(n)],
        tenant_ids=[tenant] * n,
        arrival_times=[arrival + i * 1e-3 for i in range(n)],
        **kwargs,
    )


def measure_share_deviation():
    """Two tenants (weights 2:1), B bursts; both stay backlogged."""
    scheduler = create_scheduler(
        scheduling_policy="fairshare",
        fair_share=FairShareConfig(
            tenant_weights={"A": 2.0, "B": 1.0},
            # Small window so the short measurement run settles several
            # accounting windows for the exported detail.
            window_size_seconds=0.02,
        ),
        max_num_seqs=4,
        num_blocks=10000,
    )
    for request in make_requests("A", 100, "a") + make_requests("B", 100, "b"):
        scheduler.add_request(request)
    run(scheduler, 60)
    # Settle the open accounting windows so the exported detail is complete
    # even though the measurement run is shorter than one window.
    scheduler.fair_share.maybe_advance_window(time.monotonic() + 1.0)
    snapshot = scheduler.fair_share.snapshot()
    tokens = {tenant: stats["tokens"] for tenant, stats in snapshot["tenants"].items()}
    total = tokens["A"] + tokens["B"]
    share_a = tokens["A"] / total
    report_path = scheduler.fair_share.export_report(REPORTS_DIR)
    return {
        "scenario": "two tenants, weights 2:1, both backlogged, 60 steps",
        "tokens_charged": tokens,
        "share_A": share_a,
        "expected_share_A": 2 / 3,
        "deviation": share_a - 2 / 3,
        "threshold": 0.05,
        "fairness_report": str(report_path),
    }


def measure_low_weight_wait():
    """4:1 weights; low-weight tenant's worst wait vs. the bound."""
    scheduler = create_scheduler(
        scheduling_policy="fairshare",
        fair_share=FairShareConfig(tenant_weights={"H": 4.0, "L": 1.0}),
        max_num_seqs=4,
        num_blocks=10000,
    )
    for request in make_requests("H", 100, "h"):
        scheduler.add_request(request)
    for request in make_requests("L", 8, "l"):
        scheduler.add_request(request)
        run(scheduler, 3)
    run(scheduler, 10000)
    wait = scheduler.fair_share.snapshot()["tenants"]["L"]["wait"]
    bound = 2 * (1 + 5) * 4
    return {
        "scenario": "weights 4:1, H bursting, 8 low-weight requests",
        "low_tenant_wait_rounds_max": wait["max_rounds"],
        "bound_rounds": bound,
        "low_tenant_wait_seconds_max": wait["max_seconds"],
        "low_tenant_requests_completed": wait["count"],
    }


def measure_replay_determinism():
    def once():
        scheduler = create_scheduler(
            scheduling_policy="fairshare",
            fair_share=FairShareConfig(
                tenant_weights={"A": 3.0, "B": 1.0, "C": 2.0},
                priority_aging_seconds=2.0,
            ),
            max_num_seqs=4,
            num_blocks=10000,
        )
        specs = [
            ("A", 12, 0, 0.0),
            ("B", 8, 0, 0.5),
            ("C", 10, 1, 0.25),
            ("A", 6, -1, 2.0),
            ("B", 6, 3, 2.5),
        ]
        for tenant, n, priority, base in specs:
            for request in make_requests(
                tenant,
                n,
                f"{tenant}{priority}",
                arrival=base,
                priorities=[priority] * n,
            ):
                scheduler.add_request(request)
        return run(scheduler, 50)

    first, second = once(), once()
    return {
        "scenario": "3 tenants, mixed priorities, fixed arrival sequence",
        "steps": len(first),
        "identical": first == second,
    }


def measure_single_tenant_throughput():
    def once(policy):
        fair_share = (
            FairShareConfig(tenant_weights={"only": 1.0})
            if policy == "fairshare"
            else None
        )
        scheduler = create_scheduler(
            scheduling_policy=policy,
            fair_share=fair_share,
            max_num_seqs=4,
            num_blocks=10000,
        )
        for request in make_requests("only", 60, "r"):
            scheduler.add_request(request)
        start = time.perf_counter()
        scheduled = run(scheduler, 10000)
        elapsed = time.perf_counter() - start
        tokens = sum(1 for step in scheduled for _ in step)
        return scheduled, tokens, elapsed

    fcfs_sched, fcfs_tokens, fcfs_time = once("fcfs")
    fair_sched, fair_tokens, fair_time = once("fairshare")
    return {
        "scenario": "single tenant, 60 identical requests",
        "fcfs": {
            "steps_to_drain": len(fcfs_sched),
            "scheduled_tokens": fcfs_tokens,
            "wall_seconds": fcfs_time,
        },
        "fairshare": {
            "steps_to_drain": len(fair_sched),
            "scheduled_tokens": fair_tokens,
            "wall_seconds": fair_time,
        },
        "schedule_sequence_identical": fcfs_sched == fair_sched,
        "scheduling_overhead_ratio": (fair_time / fcfs_time if fcfs_time else None),
    }


def main():
    measurements = {
        "share_deviation": measure_share_deviation(),
        "low_weight_wait": measure_low_weight_wait(),
        "replay_determinism": measure_replay_determinism(),
        "single_tenant_throughput": measure_single_tenant_throughput(),
    }
    out = REPORTS_DIR / "measurements.json"
    out.write_text(json.dumps(measurements, indent=2) + "\n")
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
