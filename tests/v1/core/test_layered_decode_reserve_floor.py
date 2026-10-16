# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests of the actual scheduler reserve/parser methods."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def scheduler_class():
    path = Path(__file__).parents[3] / "vllm/v1/core/sched/scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    cls.bases = []
    wanted = {
        "_read_layered_decode_reserve_tokens",
        "_reserve_layered_decode_budget",
        "_is_layered_decode_request",
        "_layered_pp_decode_reserve",
    }
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
    namespace = dict(os=os, Request=NS)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Scheduler"]


def setup_case(cls, *, floor=0, budget=8448, query=None, groups=2, pp=2, tp=2):
    scheduler = cls()
    scheduler.use_v2_model_runner = True
    scheduler.parallel_config = NS(pipeline_parallel_size=pp, tensor_parallel_size=tp)
    scheduler.parallel_config.data_parallel_size = 1
    scheduler.use_pp = pp > 1
    scheduler.num_spec_tokens = 0
    scheduler.max_num_running_reqs = 1
    scheduler.scheduler_config = NS(
        async_scheduling=False, long_prefill_token_threshold=0
    )
    scheduler.current_step = 0
    scheduler.max_model_len = 32768
    scheduler.max_num_scheduled_tokens = budget
    scheduler.num_sampled_tokens_per_step = 1
    scheduler._layered_decode_reserve_tokens = floor
    scheduler._is_layered_prefill_pending = lambda req: req.pending
    scheduler.running = []
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    candidate = NS(
        layered_prefill_query_tokens=budget if query is None else query,
        layered_prefill_group_id=0,
        num_in_flight_tokens=0,
        layered_prefill_num_groups=groups,
    )
    return scheduler, candidate


def decode(*, demand=1, inflight=0, prefill=False):
    return NS(
        pending=False,
        output_token_ids=[] if prefill else [1000],
        num_computed_tokens=8,
        num_tokens_with_spec=8 + demand,
        num_prompt_tokens=8,
        num_in_flight_tokens=inflight,
        num_output_placeholders=0,
        next_decode_eligible_step=0,
        max_tokens=256,
        spec_token_ids=[],
    )


@pytest.mark.parametrize("raw, expected", [(None, 0), ("0", 0), ("64", 64), (" 2 ", 2)])
def test_floor_parser(scheduler_class, monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("VLLM_LAYERED_DECODE_RESERVE_TOKENS", raising=False)
    else:
        monkeypatch.setenv("VLLM_LAYERED_DECODE_RESERVE_TOKENS", raw)
    assert scheduler_class._read_layered_decode_reserve_tokens() == expected


@pytest.mark.parametrize("raw", ["-1", "x", "1.5", ""])
def test_floor_parser_rejects_invalid_values(scheduler_class, monkeypatch, raw):
    monkeypatch.setenv("VLLM_LAYERED_DECODE_RESERVE_TOKENS", raw)
    with pytest.raises(ValueError, match="non-negative integer"):
        scheduler_class._read_layered_decode_reserve_tokens()


@pytest.mark.parametrize(
    "budget, query, floor, tp, demand, expected",
    [
        (8448, 8448, 0, 2, 0, 8448),
        (8448, 8448, 0, 2, 1, 8446),
        (8448, 8448, 64, 2, 0, 8384),
        (4224, 4224, 64, 2, 0, 4160),
        (16448, 16380, 64, 2, 0, 16380),
        (16, 16, 3, 4, 0, 12),
        (16, 16, 3, 4, 5, 8),
        (16, 3, 4, 2, 0, 3),
        (4, 4, 99, 2, 0, 1),
        (1, 1, 64, 2, 0, 1),
    ],
)
def test_floor_alignment_actual_demand_and_range_clamp(
    scheduler_class,
    budget,
    query,
    floor,
    tp,
    demand,
    expected,
):
    scheduler, candidate = setup_case(
        scheduler_class, floor=floor, budget=budget, query=query, tp=tp
    )
    if demand:
        scheduler.running = [decode(demand=demand)]
    assert scheduler._reserve_layered_decode_budget(candidate) == expected
    assert candidate.layered_prefill_query_tokens == expected


@pytest.mark.parametrize("pp, groups", [(1, 2), (4, 2), (2, 1)])
@pytest.mark.parametrize("demand", [0, 1])
def test_unsupported_topologies_keep_original_actual_reservation(
    scheduler_class, pp, groups, demand
):
    scheduler, candidate = setup_case(
        scheduler_class, floor=64, budget=16, pp=pp, groups=groups
    )
    if demand:
        scheduler.running = [decode(demand=demand)]
    expected = 14 if pp > 1 and demand else 16
    assert scheduler._reserve_layered_decode_budget(candidate) == expected


@pytest.mark.parametrize("blocked", ["legacy", "async", "later_group", "inflight"])
def test_floor_never_resizes_unsupported_or_inflight_chunks(scheduler_class, blocked):
    scheduler, candidate = setup_case(scheduler_class, floor=64, budget=16)
    scheduler.running = [decode()]
    if blocked == "legacy":
        scheduler.use_v2_model_runner = False
    elif blocked == "async":
        scheduler.scheduler_config.async_scheduling = True
    elif blocked == "later_group":
        candidate.layered_prefill_group_id = 1
    else:
        candidate.num_in_flight_tokens = 16
    assert scheduler._reserve_layered_decode_budget(candidate) == 16


@pytest.mark.parametrize("floor, expected", [(0, 8), (2, 6)])
def test_pending_decode_and_first_sample_become_ready_after_width_freezes(
    scheduler_class, floor, expected
):
    scheduler, candidate = setup_case(scheduler_class, floor=floor, budget=8)
    a = decode(demand=0, inflight=1)
    b = decode(demand=0, inflight=3, prefill=True)
    scheduler.running = [a, b]
    assert scheduler._reserve_layered_decode_budget(candidate) == expected
    # Retire B's final P RPC, which also carried A's previous Decode token.
    a.num_in_flight_tokens = b.num_in_flight_tokens = 0
    a.num_tokens_with_spec += 1
    b.num_tokens_with_spec += 1
    b.output_token_ids = [1000]
    candidate.layered_prefill_group_id = 1
    candidate.num_in_flight_tokens = expected
    assert scheduler._reserve_layered_decode_budget(candidate) == expected
    assert scheduler.max_num_scheduled_tokens - expected == (2 if floor else 0)


@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16])
@pytest.mark.parametrize("floor, expected", [(0, 12), (2, 12), (6, 10)])
def test_automatic_reserve_covers_every_group_layout(
    scheduler_class,
    groups,
    floor,
    expected,
):
    scheduler, candidate = setup_case(
        scheduler_class,
        groups=groups,
        floor=floor,
        budget=16,
    )
    scheduler.max_num_running_reqs = 4
    scheduler.running = [decode(demand=0, inflight=1)]
    assert scheduler._reserve_layered_decode_budget(candidate) == expected
    for group in range(1, groups):
        candidate.layered_prefill_group_id = group
        candidate.num_in_flight_tokens = expected
        scheduler.running = [decode(demand=5)]
        assert scheduler._reserve_layered_decode_budget(candidate) == expected


@pytest.mark.parametrize("groups", [2, 8, 16])
@pytest.mark.parametrize(
    "blocked",
    ["legacy", "async", "small_budget", "recompute"],
)
def test_automatic_reserve_retains_fallback_guards(scheduler_class, groups, blocked):
    scheduler, candidate = setup_case(scheduler_class, groups=groups, budget=16)
    scheduler.max_num_running_reqs = 4
    scheduler.running = [decode(demand=0, inflight=1)]
    if blocked == "legacy":
        scheduler.use_v2_model_runner = False
    elif blocked == "async":
        scheduler.scheduler_config.async_scheduling = True
    elif blocked == "small_budget":
        scheduler.max_num_scheduled_tokens = 4
    else:
        scheduler.running = [decode(demand=2)]
    assert scheduler._layered_pp_decode_reserve(candidate) == 0


@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16])
def test_automatic_reserve_keeps_single_prefill_full_width(scheduler_class, groups):
    scheduler, candidate = setup_case(scheduler_class, groups=groups, budget=16)
    scheduler.max_num_running_reqs = 4
    assert scheduler._reserve_layered_decode_budget(candidate) == 16
