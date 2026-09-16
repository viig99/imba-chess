"""Profiling separates launch work from waits and records comparable starts."""

from types import SimpleNamespace

import torch

from scripts.profile_gumbel_pipeline import preparation_attribution


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
    assert result["fraction"] == 0.11
    assert result["threshold_met"]
    assert not preparation_attribution([])["threshold_met"]
