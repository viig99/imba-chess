"""Promotion evidence must reject incomplete, regressing or still-compiling runs."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts.profile_gumbel_pipeline import preparation_attribution, promotion_report


def reports():
    result = {}
    for i in range(3):
        for mode in ("current", "direct"):
            result[f"pair_{i}_{mode}"] = dict(
                usable_positions_per_hour=100 if mode == "current" else 107,
                move_latency_p95=1,
                peak_allocated_bytes=3 * 1024**3,
                compiler_before={"stats": {"unique_graphs": 2}},
                compiler_after={"stats": {"unique_graphs": 2}},
                allocated_after_collection=200_000_000,
                cache_empty_after_collection=True,
            )
    return result


def gate(data, games=128):
    return promotion_report(data, games, 3, "current", "direct")


def test_promotion_requires_full_workload_and_each_pair_improves():
    data = reports()
    assert gate(data)["passes"]
    assert not gate(data, games=32)["passes"]
    data["pair_1_direct"]["usable_positions_per_hour"] = 99
    result = gate(data)
    assert result["checks"]["median_throughput"]
    assert not result["checks"]["every_pair"]
    assert not result["passes"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("move_latency_p95", 1.06),
        ("peak_allocated_bytes", 6 * 1024**3),
        ("compiler_after", {"stats": {"unique_graphs": 3}}),
        ("allocated_after_collection", 202_000_000),
        ("cache_empty_after_collection", False),
    ],
)
def test_promotion_rejects_latency_memory_compilation_and_retention(key, value):
    data = deepcopy(reports())
    for i in range(3):
        data[f"pair_{i}_direct"][key] = value
    assert not gate(data)["passes"]


def test_fusion_attribution_excludes_waits_and_neural_execution():
    def event(name, own, total=0, parent=None, device=torch.autograd.DeviceType.CPU):
        return SimpleNamespace(
            name=name,
            self_cpu_time_total=own,
            cpu_time_total=total,
            cpu_parent=parent,
            device_type=device,
        )

    root = event("gumbel_searches", 0, 1000)
    prep = event("preparation_ancestor_gather", 20, 600, root)
    op = event("aten::gather", 40, 580, prep)
    launch = event("cudaLaunchKernel", 50, 50, op)
    wait = event("cudaStreamSynchronize", 490, 490, op)
    gpu = event("gather_kernel", 0, 0, op, torch.autograd.DeviceType.CUDA)
    decoder = event("decoder_execution", 300, 300, root)
    result = preparation_attribution([root, prep, op, launch, wait, gpu, decoder])
    assert result["exclusive_cpu_launch_us"] == 110
    assert result["fraction"] == pytest.approx(0.11)
    assert result["threshold_met"]
    assert not preparation_attribution([])["threshold_met"]


def test_thermal_start_requires_cool_cpu_gpu_and_no_throttling():
    from scripts.profile_gumbel_pipeline import thermal_ready

    ready = dict(
        gpu_celsius=55,
        cpu_celsius=65,
        sw_thermal_slowdown="Not Active",
        hw_thermal_slowdown="Not Active",
    )
    assert thermal_ready(ready)
    for key, value in (
        ("gpu_celsius", 84),
        ("cpu_celsius", 96),
        ("cpu_celsius", None),
        ("sw_thermal_slowdown", "Active"),
        ("hw_thermal_slowdown", "Active"),
    ):
        assert not thermal_ready(dict(ready, **{key: value}))
