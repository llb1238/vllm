# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure pinned H2D copies using rotating buffers and CUDA events."""

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

BLOCK_BYTES = 2359296
BLOCK_COUNTS = (1, 16, 32, 64, 128, 625)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.accelerator.set_device_index(0)
    device = torch.cuda.get_device_properties(0)
    ring_bytes = 512 * 1024**2
    cases = []
    total_bytes = max(
        n * BLOCK_BYTES * max(1, math.ceil(ring_bytes / (n * BLOCK_BYTES)))
        for n in BLOCK_COUNTS
    )
    host = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    host.random_(0, 256, generator=torch.Generator().manual_seed(0))
    gpu = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
    assert host.is_pinned()
    host_address = host.data_ptr()
    mapping_start = None
    for line in Path("/proc/self/maps").read_text().splitlines():
        left, right = (int(x, 16) for x in line.split()[0].split("-"))
        if left <= host_address < right:
            mapping_start = left
            break
    numa_mapping = [
        line
        for line in Path("/proc/self/numa_maps").read_text().splitlines()
        if mapping_start is not None and line.split()[0] == format(mapping_start, "x")
    ]
    metadata = {
        "gpu": device.name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "host_pinned": host.is_pinned(),
        "host_pool_bytes": total_bytes,
        "host_numa_mapping": numa_mapping,
        "ring_target_bytes": ring_bytes,
        "rounds": args.rounds,
        "samples_per_round": args.repeats,
        "warmup": args.warmup,
        "timing": "CUDA events; CUPTI unavailable; allocation and correctness excluded",
        "graph_method": "different host/device views per node; >=512 MiB footprint",
        "nvidia_smi_before": subprocess.check_output(["nvidia-smi"], text=True),
    }
    (args.output_dir / "bandwidth-environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    stream = torch.cuda.Stream()
    for blocks in BLOCK_COUNTS:
        size = blocks * BLOCK_BYTES
        slots = max(1, math.ceil(ring_bytes / size))
        src = [host[i * size : (i + 1) * size] for i in range(slots)]
        dst = [gpu[i * size : (i + 1) * size] for i in range(slots)]
        with torch.cuda.stream(stream):
            dst[0].copy_(src[0], non_blocking=True)
        stream.synchronize()
        for offset in range(0, size, 64 * 1024**2):
            end = min(size, offset + 64 * 1024**2)
            assert torch.equal(gpu[offset:end].cpu(), host[offset:end])
        # Warm up before graph capture; all views and memory are preallocated.
        with torch.cuda.stream(stream):
            for a, z in zip(src, dst):
                z.copy_(a, non_blocking=True)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for a, z in zip(src, dst):
                z.copy_(a, non_blocking=True)
        for _ in range(args.warmup):
            with torch.cuda.stream(stream):
                graph.replay()
        stream.synchronize()
        cases.append(
            {
                "blocks": blocks,
                "bytes": size,
                "slots": slots,
                "src": src,
                "dst": dst,
                "graph": graph,
            }
        )
        print(f"CORRECT bytes={size} blocks={blocks} ring_slots={slots}", flush=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for round_index in range(args.rounds):
        order = cases[round_index:] + cases[:round_index]
        if round_index % 2:
            order = list(reversed(order))
        for case in order:
            for iteration in range(args.repeats):
                slot = iteration % case["slots"]
                methods = ("direct", "graph")
                if (iteration + round_index) % 2:
                    methods = tuple(reversed(methods))
                for method in methods:
                    stream.synchronize()
                    wall_start = time.perf_counter()
                    with torch.cuda.stream(stream):
                        start.record(stream)
                        if method == "direct":
                            case["dst"][slot].copy_(
                                case["src"][slot], non_blocking=True
                            )
                        else:
                            case["graph"].replay()
                        end.record(stream)
                    end.synchronize()
                    divisor = case["slots"] if method == "graph" else 1
                    ms = start.elapsed_time(end) / divisor
                    wall_ms = (time.perf_counter() - wall_start) * 1000 / divisor
                    samples.append(
                        {
                            "round": round_index,
                            "iteration": iteration,
                            "method": method,
                            "bytes": case["bytes"],
                            "blocks": case["blocks"],
                            "batch_copies": divisor,
                            "cuda_ms": ms,
                            "wall_ms": wall_ms,
                            "gbps": case["bytes"] / (ms * 1e6),
                        }
                    )
            print(f"ROUND {round_index + 1} bytes={case['bytes']} done", flush=True)
    summary = []
    for case in cases:
        for method in ("direct", "graph"):
            data = [
                s
                for s in samples
                if s["bytes"] == case["bytes"] and s["method"] == method
            ]
            times = sorted(s["cuda_ms"] for s in data)
            median_ms = statistics.median(times)
            summary.append(
                {
                    "bytes": case["bytes"],
                    "blocks": case["blocks"],
                    "method": method,
                    "samples": len(data),
                    "median_cuda_ms": median_ms,
                    "p10_cuda_ms": times[int(len(times) * 0.1)],
                    "p90_cuda_ms": times[int(len(times) * 0.9)],
                    "median_wall_ms": statistics.median(s["wall_ms"] for s in data),
                    "gbps_from_median": case["bytes"] / (median_ms * 1e6),
                }
            )
    (args.output_dir / "bandwidth-samples.json").write_text(
        json.dumps(samples, indent=2) + "\n"
    )
    (args.output_dir / "bandwidth-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("BANDWIDTH_RESULT " + json.dumps(summary), flush=True)
    del cases, gpu, host
    gc.collect()


if __name__ == "__main__":
    main()
