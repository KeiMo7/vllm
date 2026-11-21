# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

from vllm.v1.core.batch_trace import (
    BatchTraceRecorder,
    diff_traces,
    load_records,
    summarize,
)
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

# Per-record overhead budget for the recorder (JSON build + write + flush).
# Typical values are in the tens of microseconds; this is a regression
# tripwire, not a tight bound.
MAX_AVG_RECORD_OVERHEAD_US = 500


def _read_trace_lines(directory):
    lines = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".jsonl"):
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                lines.extend(f.read().splitlines())
    return lines


def _fake_req(req_id, prompt=8, computed=0, output=0):
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
        num_output_tokens=output,
        num_tokens=prompt + output,
    )


def _record_batch(recorder, req_ids=("r0",), tokens=None):
    reqs = [_fake_req(rid) for rid in req_ids]
    num_scheduled = {rid: (tokens or 8) for rid in req_ids}
    recorder.record_batch(
        new_reqs=reqs,
        cached_reqs=[],
        num_scheduled_tokens=num_scheduled,
        num_waiting=0,
        running_reqs=reqs,
        kv_free_blocks=500,
        kv_total_blocks=1000,
        block_size=16,
        num_preempted=0,
        finished_req_ids=[],
    )


def test_tracing_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_TRACE_DIR", raising=False)
    scheduler = create_scheduler()
    assert scheduler.batch_trace_recorder is None


def test_records_batch_composition(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_BATCH_TRACE_DIR", str(tmp_path))
    scheduler = create_scheduler()
    assert scheduler.batch_trace_recorder is not None
    assert scheduler.batch_trace_recorder.enabled

    requests = create_requests(num_requests=4, num_tokens=10)
    for req in requests:
        scheduler.add_request(req)

    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens > 0
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[r.request_id for r in requests],
            sampled_token_ids=[[0] for _ in requests],
        ),
    )
    scheduler.shutdown()

    records = [json.loads(line) for line in _read_trace_lines(tmp_path)]
    assert records[0]["type"] == "meta"
    batches = [r for r in records if r["type"] == "batch"]
    step_ends = [r for r in records if r["type"] == "step_end"]
    assert len(batches) == 1
    assert len(step_ends) == 1
    assert step_ends[0]["seq"] == batches[0]["seq"]

    batch = batches[0]
    traced = {req["id"]: req for req in batch["reqs"]}
    assert set(traced) == {r.request_id for r in requests}
    for req in requests:
        entry = traced[req.request_id]
        assert entry["sched"] == output.num_scheduled_tokens[req.request_id]
        assert entry["new"] is True
        assert entry["phase"] == "prefill"
        assert entry["prompt"] == 10
    assert batch["num_reqs"] == 4
    assert batch["tokens"] == output.total_num_scheduled_tokens
    assert 0.0 <= batch["kv_usage"] <= 1.0
    assert batch["kv_total"] > 0
    assert batch["kv_slots"] >= batch["kv_tokens"]
    assert "ts" in batch and "mono" in batch


def test_trace_contains_no_prompt_content(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_BATCH_TRACE_DIR", str(tmp_path))
    scheduler = create_scheduler()
    requests = create_requests(num_requests=2, num_tokens=6)
    # Distinctive token ids that must never appear in the trace.
    for i, req in enumerate(requests):
        req.prompt_token_ids = [777_000_111 + i] * req.num_prompt_tokens
    for req in requests:
        scheduler.add_request(req)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[r.request_id for r in requests],
            sampled_token_ids=[[0] for _ in requests],
        ),
    )
    scheduler.shutdown()

    raw = "".join(_read_trace_lines(tmp_path))
    for req in requests:
        for token_id in req.prompt_token_ids:
            assert str(token_id) not in raw

    allowed_keys = {"id", "new", "phase", "sched", "prompt", "computed", "output"}
    for record in load_records(str(tmp_path)):
        if record["type"] == "batch":
            for entry in record["reqs"]:
                assert set(entry) <= allowed_keys


def test_unwritable_directory_degrades(monkeypatch, tmp_path, caplog):
    # A path that treats a regular file as a directory: makedirs always
    # fails with NotADirectoryError, even when running as root.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("VLLM_BATCH_TRACE_DIR", str(blocker / "trace"))

    with caplog.at_level("WARNING"):
        scheduler = create_scheduler()
    recorder = scheduler.batch_trace_recorder
    assert recorder is not None
    assert not recorder.enabled
    assert any("not writable" in m for m in caplog.messages)

    # Inference path is unaffected: scheduling and output processing work.
    requests = create_requests(num_requests=2, num_tokens=10)
    for req in requests:
        scheduler.add_request(req)
    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens > 0
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[r.request_id for r in requests],
            sampled_token_ids=[[0] for _ in requests],
        ),
    )
    scheduler.shutdown()


def test_records_survive_sigkill(tmp_path):
    trace_dir = tmp_path / "trace"
    program = (
        "import os, signal;"
        "from vllm.v1.core.batch_trace import BatchTraceRecorder;"
        "from types import SimpleNamespace as NS;"
        f"r = BatchTraceRecorder({str(trace_dir)!r});"
        "reqs = [NS(request_id='r0', num_prompt_tokens=8,"
        " num_computed_tokens=0, num_output_tokens=0, num_tokens=8)];"
        "[r.record_batch(new_reqs=reqs, cached_reqs=[],"
        " num_scheduled_tokens={'r0': 8}, num_waiting=0, running_reqs=reqs,"
        " kv_free_blocks=1, kv_total_blocks=2, block_size=16,"
        " num_preempted=0, finished_req_ids=[]) for _ in range(50)];"
        "os.kill(os.getpid(), signal.SIGKILL)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stderr.decode()

    records = [json.loads(line) for line in _read_trace_lines(trace_dir)]
    batches = [r for r in records if r["type"] == "batch"]
    # Every record produced before the kill is complete and parseable.
    assert len(batches) == 50
    assert [b["seq"] for b in batches] == list(range(50))


def test_record_overhead_below_threshold(tmp_path):
    recorder = BatchTraceRecorder(str(tmp_path))
    iterations = 200
    for _ in range(iterations):
        _record_batch(recorder, req_ids=tuple(f"r{j}" for j in range(16)))
        recorder.record_step_end()
    recorder.close()

    avg_us = recorder.avg_record_overhead_ns / 1e3
    assert recorder.records_written >= iterations * 2
    assert recorder.records_dropped == 0
    assert avg_us < MAX_AVG_RECORD_OVERHEAD_US, (
        f"average record overhead {avg_us:.1f}us exceeds {MAX_AVG_RECORD_OVERHEAD_US}us"
    )


def test_rotation_by_size(tmp_path):
    recorder = BatchTraceRecorder(str(tmp_path), max_file_bytes=4096)
    for _ in range(100):
        _record_batch(recorder)
    recorder.close()

    files = [f for f in os.listdir(tmp_path) if f.endswith(".jsonl")]
    assert len(files) > 1
    for f in files:
        assert os.path.getsize(tmp_path / f) > 0
    # Every line in every segment is a complete JSON record.
    for line in _read_trace_lines(tmp_path):
        json.loads(line)
    seqs = [r["seq"] for r in load_records(str(tmp_path)) if r["type"] == "batch"]
    assert seqs == list(range(100))


def test_single_record_failure_is_counted(tmp_path):
    recorder = BatchTraceRecorder(str(tmp_path))
    recorder._file.close()
    # A closed file raises ValueError on write; simulate an I/O failure by
    # pointing the recorder at a file object that raises OSError.
    recorder._file = _FailingFile()
    _record_batch(recorder)
    assert recorder.records_dropped == 1
    assert recorder.enabled  # single failure does not disable tracing
    recorder._file = None  # avoid close() touching the fake


class _FailingFile:
    def write(self, _line):
        raise OSError("disk full")

    def flush(self):
        pass

    def close(self):
        pass


def test_summarize_and_diff(tmp_path):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    rec_a = BatchTraceRecorder(str(dir_a))
    rec_b = BatchTraceRecorder(str(dir_b))
    # Identical second batch, diverging first batch and admission order.
    for rec, first in ((rec_a, ("r0", "r1")), (rec_b, ("r0", "r2"))):
        _record_batch(rec, req_ids=first)
        rec.record_step_end()
        _record_batch(rec, req_ids=("r1", "r2"))
        rec.record_step_end()
        rec.close()

    records_a = load_records(str(dir_a))
    records_b = load_records(str(dir_b))

    summary = summarize(records_a)
    assert summary["num_batches"] == 2
    assert summary["batch_size_reqs"]["avg"] == 2
    assert summary["step_duration_ms"] is not None
    assert summary["kv_waste_ratio"] is not None

    report = diff_traces(records_a, records_b)
    assert report["num_differing_batches"] == 1
    divergence = report["first_divergence"]
    assert divergence["index"] == 0
    assert divergence["only_in_a"] == ["r1"]
    assert divergence["only_in_b"] == ["r2"]
    # r1 admitted one step later in run B, r2 one step earlier.
    assert report["admission_order_deltas"] == {"r1": 1, "r2": -1}
