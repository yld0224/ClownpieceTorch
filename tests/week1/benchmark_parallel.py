#!/usr/bin/env python3
"""Build and benchmark every tensor parallel backend in isolated processes."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
CLOWNPIECE_DIR = ROOT / "clownpiece"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "parallel_results"
BACKENDS = ("serial", "stdthread", "openmp", "threadpool")
BACKEND_LABELS = {
    "serial": "Serial",
    "stdthread": "std::thread per operation",
    "openmp": "OpenMP",
    "threadpool": "Thread pool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument(
        "--element-sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000, 5_000_000]
    )
    parser.add_argument("--matmul-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--matmul-repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.warmup < 0 or args.repeats <= 0 or args.matmul_repeats <= 0:
        raise ValueError("warmup must be non-negative and repeat counts must be positive")
    if any(size <= 0 for size in args.element_sizes + args.matmul_sizes):
        raise ValueError("all benchmark sizes must be positive")


def measure(
    operation: Callable[[], object],
    validate: Callable[[object], None],
    warmup: int,
    repeats: int,
) -> list[int]:
    for _ in range(warmup):
        operation()
    result = operation()
    validate(result)
    del result

    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    samples = []
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = operation()
            samples.append(time.perf_counter_ns() - start)
            del result
    finally:
        if gc_was_enabled:
            gc.enable()
    return samples


def worker_main(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(CLOWNPIECE_DIR))
    import tensor_impl as cp

    rows = []
    for size in args.element_sizes:
        tensor = cp.ones([size])

        def elementwise_operation():
            return tensor * 2.0 + 3.0

        def validate_elementwise(result):
            if result.data_at(0) != 5.0 or result.data_at(size - 1) != 5.0:
                raise RuntimeError("element-wise result validation failed")

        samples = measure(
            elementwise_operation, validate_elementwise, args.warmup, args.repeats
        )
        rows.extend(
            {
                "backend": args.backend,
                "workload": "elementwise",
                "size": size,
                "repeat": repeat,
                "time_ns": elapsed,
            }
            for repeat, elapsed in enumerate(samples)
        )
        del tensor

    for size in args.matmul_sizes:
        lhs = cp.ones([size, size])
        rhs = cp.ones([size, size])

        def matmul_operation():
            return lhs @ rhs

        def validate_matmul(result):
            expected = float(size)
            if result.data_at(0) != expected or result.data_at(size * size - 1) != expected:
                raise RuntimeError("matrix multiplication result validation failed")

        samples = measure(
            matmul_operation, validate_matmul, args.warmup, args.matmul_repeats
        )
        rows.extend(
            {
                "backend": args.backend,
                "workload": "matmul",
                "size": size,
                "repeat": repeat,
                "time_ns": elapsed,
            }
            for repeat, elapsed in enumerate(samples)
        )
        del lhs, rhs

    print(json.dumps(rows))


def build_backend(backend: str, env: dict[str, str]) -> None:
    build_env = env.copy()
    build_env["CP_PARALLEL_MODE"] = backend
    command = [sys.executable, "setup.py", "build_ext", "--inplace", "--force"]
    result = subprocess.run(
        command,
        cwd=CLOWNPIECE_DIR,
        env=build_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to build {backend} backend:\n{result.stdout}")


def benchmark_backend(backend: str, args: argparse.Namespace, env: dict[str, str]) -> list[dict]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--matmul-repeats",
        str(args.matmul_repeats),
        "--element-sizes",
        *map(str, args.element_sizes),
        "--matmul-sizes",
        *map(str, args.matmul_sizes),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"benchmark worker for {backend} failed:\n{result.stdout}\n{result.stderr}"
        )
    return json.loads(result.stdout)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[int]] = {}
    for row in rows:
        key = (row["backend"], row["workload"], row["size"])
        grouped.setdefault(key, []).append(row["time_ns"])

    medians = {key: statistics.median(values) for key, values in grouped.items()}
    summary = []
    for backend in BACKENDS:
        for workload in ("elementwise", "matmul"):
            sizes = sorted(key[2] for key in grouped if key[:2] == (backend, workload))
            for size in sizes:
                values = grouped[(backend, workload, size)]
                median_ns = medians[(backend, workload, size)]
                serial_ns = medians.get(("serial", workload, size))
                summary.append(
                    {
                        "backend": backend,
                        "workload": workload,
                        "size": size,
                        "samples": len(values),
                        "median_ms": median_ns / 1_000_000,
                        "mean_ms": statistics.fmean(values) / 1_000_000,
                        "min_ms": min(values) / 1_000_000,
                        "speedup_vs_serial": serial_ns / median_ns if serial_ns else None,
                    }
                )
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output_dir: Path, summary: list[dict], workload: str, filename: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for backend in BACKENDS:
        points = [
            row for row in summary if row["backend"] == backend and row["workload"] == workload
        ]
        axis.plot(
            [row["size"] for row in points],
            [row["median_ms"] for row in points],
            marker="o",
            linewidth=1.8,
            label=BACKEND_LABELS[backend],
        )

    axis.set_xscale("log", base=10 if workload == "elementwise" else 2)
    axis.set_yscale("log")
    axis.set_xlabel("Number of elements" if workload == "elementwise" else "Square matrix size N")
    axis.set_ylabel("Median runtime (ms)")
    axis.set_title("Element-wise: y = x * 2 + 3" if workload == "elementwise" else "Matrix multiplication: (N,N) @ (N,N)")
    axis.grid(True, which="both", linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=160)
    plt.close(figure)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def controller_main(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/clownpiece-matplotlib")
    env = os.environ.copy()
    env["CP_NUM_THREADS"] = str(args.threads)

    all_rows = []
    current_backend = None
    try:
        for backend in BACKENDS:
            print(f"[{backend}] building...", flush=True)
            build_backend(backend, env)
            current_backend = backend
            print(f"[{backend}] benchmarking...", flush=True)
            all_rows.extend(benchmark_backend(backend, args, env))
    finally:
        if current_backend != "serial":
            print("[serial] restoring default build...", flush=True)
            build_backend("serial", env)

    summary = summarize(all_rows)
    raw_rows = [
        {**row, "time_ms": row["time_ns"] / 1_000_000}
        for row in all_rows
    ]
    write_csv(
        output_dir / "parallel_raw.csv",
        raw_rows,
        ["backend", "workload", "size", "repeat", "time_ns", "time_ms"],
    )
    write_csv(
        output_dir / "parallel_summary.csv",
        summary,
        [
            "backend",
            "workload",
            "size",
            "samples",
            "median_ms",
            "mean_ms",
            "min_ms",
            "speedup_vs_serial",
        ],
    )
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "configured_threads": args.threads,
        "python": sys.version.split()[0],
        "element_sizes": args.element_sizes,
        "matmul_sizes": args.matmul_sizes,
        "warmup": args.warmup,
        "element_repeats": args.repeats,
        "matmul_repeats": args.matmul_repeats,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    plot_results(output_dir, summary, "elementwise", "elementwise_runtime.png")
    plot_results(output_dir, summary, "matmul", "matmul_runtime.png")
    print(f"Results written to {output_dir}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.worker:
        if args.backend is None:
            raise ValueError("--worker requires --backend")
        worker_main(args)
    else:
        controller_main(args)


if __name__ == "__main__":
    main()
