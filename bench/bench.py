"""Benchmarks against upstream jieba on identical Chinese text."""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import jieba as upstream  # noqa: E402
import mojo_jieba as mojo  # noqa: E402


def timeit(function, repeat=5):
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), result


def machine() -> str:
    model = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model or platform.machine()}, {platform.system()} {platform.release()}"


def main():
    unit = (
        "南京市长江大桥欢迎您。"
        "小明硕士毕业于中国科学院计算所，后在日本京都大学深造。"
        "他来到了网易杭研大厦。"
    )
    text = unit * 4_000
    mojo.initialize()
    upstream.initialize()
    mojo.lcut(unit, HMM=False)
    upstream.lcut(unit, HMM=False)

    cases = [
        (
            f"accurate, HMM=False ({len(text):,} chars)",
            lambda: mojo.lcut(text, HMM=False),
            lambda: upstream.lcut(text, HMM=False),
        ),
        (
            f"accurate, HMM=True ({len(text):,} chars)",
            lambda: mojo.lcut(text, HMM=True),
            lambda: upstream.lcut(text, HMM=True),
        ),
        (
            f"search, HMM=False ({len(text):,} chars)",
            lambda: mojo.lcut_for_search(text, HMM=False),
            lambda: upstream.lcut_for_search(text, HMM=False),
        ),
        (
            f"tokenize search ({len(text):,} chars)",
            lambda: list(mojo.tokenize(text, mode="search", HMM=False)),
            lambda: list(upstream.tokenize(text, mode="search", HMM=False)),
        ),
    ]

    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-jieba | jieba 0.42.1 | speedup |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases:
        ours_time, ours_result = timeit(ours)
        theirs_time, theirs_result = timeit(theirs)
        if ours_result != theirs_result:
            raise AssertionError(f"benchmark outputs differ for {name}")
        print(
            f"| {name} | {ours_time * 1000:.2f} ms | "
            f"{theirs_time * 1000:.2f} ms | {theirs_time / ours_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
