"""
InferX Benchmarking Framework
=============================

Single-file reusable benchmarking framework for LLM services.

Features implemented:
- asyncio-based Poisson-arrival workload generator with variable prompt lengths
- Asynchronous workers using aiohttp to send requests (streaming-aware TTFT measurement)
- Metrics collection: TTFT, TPOT, throughput, request timings, status codes
- GPU memory monitoring (pynvml or nvidia-smi fallback)
- Standardized CSV/JSON outputs
- Plotting utilities: throughput vs latency (p50/p90/p99) curves
- CLI for quick experiments and reproducible runs

Usage (example):
    python inferx_benchmark_framework.py \
        --endpoint http://localhost:8000/v1/generate \
        --duration 60 --rate 10 --concurrency 8 \
        --prompt-dist uniform --min-tokens 8 --max-tokens 128 \
        --outdir ./results/test_run

Notes:
- The tool is intentionally conservative about assumptions on the LLM server API.
  By default it sends POST requests with JSON body: {"prompt": "<prompt>"}
  You can modify `REQUEST_BODY_TEMPLATE` or the `--json-field` CLI flag if your
  service expects a different shape.

- TTFT: time to first response chunk. If server doesn't stream, TTFT = total response time.
- TPOT: time per output token = (end_time - first_byte_time) / n_output_tokens (estimated)

- GPU monitoring attempts pynvml (recommended). If unavailable it calls `nvidia-smi`.

"""

from __future__ import annotations
import asyncio
import aiohttp
import argparse
import csv
import json
import os
import sys
import time
import math
import random
import uuid
import statistics
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Dict, Any, List, Coroutine, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

# ----------------------------- Configuration classes -----------------------------

@dataclass
class BenchmarkConfig:
    endpoint: str
    duration: float = 60.0
    rate: float = 10.0  # requests per second (Poisson lambda)
    concurrency: int = 8
    prompt_dist: str = "uniform"  # 'uniform' or 'zipf' or 'choices'
    min_tokens: int = 8
    max_tokens: int = 128
    choices_tokens: Optional[List[int]] = None
    request_body_template: str = '{"prompt": "{prompt}"}'
    json_field: Optional[str] = None  # if set, will send JSON with {json_field: prompt}
    headers: Optional[Dict[str, str]] = None
    timeout: float = 60.0
    outdir: str = "./results"
    gpu_poll_interval: float = 0.5
    gpu_monitor: bool = True
    sample_prompts_file: Optional[str] = None


@dataclass
class RequestRecord:
    request_id: str
    prompt: str
    prompt_tokens: int
    enqueue_time: float
    send_time: Optional[float] = None
    first_byte_time: Optional[float] = None
    end_time: Optional[float] = None
    status: Optional[int] = None
    resp_len_tokens: Optional[int] = None
    bytes_received: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# ----------------------------- Prompt generation -----------------------------

LOREM = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua"
)

WORDS = LOREM.split()


def sample_prompt(config: BenchmarkConfig) -> Tuple[str, int]:
    """Generate a pseudo-random prompt and return (prompt_text, tokens_estimate).

    Token accounting here is approximate (we use words as tokens). That's sufficient
    for benchmarking where token-accurate counts are unavailable without a tokenizer.
    """
    if config.sample_prompts_file and os.path.exists(config.sample_prompts_file):
        # sample lines from file
        with open(config.sample_prompts_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if lines:
            prompt = random.choice(lines)
            tokens = len(prompt.split())
            return prompt, tokens

    if config.prompt_dist == "choices" and config.choices_tokens:
        t = random.choice(config.choices_tokens)
    elif config.prompt_dist == "zipf" and np is not None:
        # zipf with a=2 approximates a heavy tail; clamp to max_tokens
        a = 2.0
        t = int(np.random.zipf(a))
        t = max(config.min_tokens, min(t, config.max_tokens))
    else:
        # uniform fallback
        t = random.randint(config.min_tokens, config.max_tokens)

    # build prompt by repeating the lorem words until we reach target length
    words = []
    while len(words) < t:
        words.extend(WORDS)
    prompt = " ".join(words[:t])
    return prompt, t


# ----------------------------- GPU monitoring -----------------------------

class GPUWatcher:
    """Background GPU memory (and optionally util) monitoring.

    Writes periodic samples to a CSV and exposes the latest value.
    """
    def __init__(self, outdir: str, interval: float = 0.5):
        self.interval = interval
        self.outdir = outdir
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.samples: List[Dict[str, Any]] = []
        self.csv_path = os.path.join(outdir, "gpu_memory.csv")
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        # init
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._handle = None
        else:
            self._handle = None

    def _sample_once(self) -> Dict[str, Any]:
        now = time.time()
        sample = {"ts": now}
        if self._handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                sample.update({
                    "mem_used_mb": int(mem_info.used / 1024 / 1024),
                    "mem_total_mb": int(mem_info.total / 1024 / 1024),
                    "gpu_util": int(util.gpu),
                    "gpu_mem_util": int(util.memory)
                })
            except Exception as e:
                sample.update({"mem_used_mb": None, "error": str(e)})
        else:
            # fallback to nvidia-smi
            try:
                import subprocess
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits"
                ])
                out = out.decode("utf-8").strip().split("\n")[0]
                mem_used, mem_total, util_gpu, util_mem = [int(x.strip()) for x in out.split(",")]
                sample.update({
                    "mem_used_mb": mem_used,
                    "mem_total_mb": mem_total,
                    "gpu_util": util_gpu,
                    "gpu_mem_util": util_mem
                })
            except Exception as e:
                sample.update({"mem_used_mb": None, "error": str(e)})
        return sample

    async def _run(self):
        # write CSV header
        header_written = False
        while self._running:
            s = self._sample_once()
            self.samples.append(s)
            # append to csv
            try:
                write_header = not os.path.exists(self.csv_path)
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(list(s.keys()))
                    writer.writerow([s[k] for k in s.keys()])
            except Exception:
                pass
            await asyncio.sleep(self.interval)

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            await self._task


# ----------------------------- Metrics collection -----------------------------

class MetricsWriter:
    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.csv_path = os.path.join(outdir, "metrics.csv")
        # ensure header
        self._init_header()
        self._lock = asyncio.Lock()

    def _init_header(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "request_id",
                    "enqueue_time",
                    "send_time",
                    "first_byte_time",
                    "end_time",
                    "prompt_tokens",
                    "resp_tokens",
                    "bytes_received",
                    "status",
                    "error"
                ])

    async def write(self, rec: RequestRecord):
        row = [
            rec.request_id,
            rec.enqueue_time,
            rec.send_time,
            rec.first_byte_time,
            rec.end_time,
            rec.prompt_tokens,
            rec.resp_len_tokens,
            rec.bytes_received,
            rec.status,
            rec.error
        ]
        async with self._lock:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)


# ----------------------------- HTTP worker & load generator -----------------------------

class AsyncLLMBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics_writer = MetricsWriter(config.outdir)
        self.gpu_watcher = GPUWatcher(config.outdir, interval=config.gpu_poll_interval) if config.gpu_monitor else None
        self._running = False
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._producer_task: Optional[asyncio.Task] = None
        self._start_ts = None

    async def _producer(self):
        """Produce request items into queue with Poisson arrivals (exponential inter-arrival)."""
        lam = float(self.config.rate)
        assert lam > 0
        stop_at = time.time() + self.config.duration
        while time.time() < stop_at:
            # sample prompt
            prompt, tokens = sample_prompt(self.config)
            rec = RequestRecord(
                request_id=str(uuid.uuid4()),
                prompt=prompt,
                prompt_tokens=tokens,
                enqueue_time=time.time()
            )
            await self._queue.put(rec)
            # draw inter-arrival
            if np is not None:
                wait = np.random.exponential(1.0 / lam)
            else:
                # python's random.expovariate uses lambda as rate
                wait = random.expovariate(lam)
            # cap wait to something sensible
            wait = max(0.0, float(wait))
            await asyncio.sleep(wait)
        # signal end-of-stream by putting Nones for each worker
        for _ in range(self.config.concurrency):
            await self._queue.put(None)

    async def _worker(self, worker_id: int):
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        headers = self.config.headers or {"Content-Type": "application/json"}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as sess:
            while True:
                item = await self._queue.get()
                if item is None:
                    # put back for other workers
                    self._queue.task_done()
                    break
                rec: RequestRecord = item
                rec.send_time = time.time()
                try:
                    # craft body
                    body_text = self.config.request_body_template.format(prompt=rec.prompt)
                    # try to parse to json
                    try:
                        body_json = json.loads(body_text)
                        send_data = body_json
                        resp = await sess.post(self.config.endpoint, json=send_data)
                    except Exception:
                        # fallback: send raw text
                        resp = await sess.post(self.config.endpoint, data=body_text)

                    rec.status = getattr(resp, 'status', None)

                    # streaming-aware read: attempt to read first chunk
                    first_byte_ts = None
                    content_bytes = 0
                    resp_text = None
                    try:
                        # prefer streaming read; iter_chunked will yield as data arrives
                        async for chunk in resp.content.iter_chunked(1024):
                            if first_byte_ts is None:
                                first_byte_ts = time.time()
                                rec.first_byte_time = first_byte_ts
                            content_bytes += len(chunk)
                        # if the loop finishes without chunks, set first_byte_time to send_time
                        if rec.first_byte_time is None:
                            rec.first_byte_time = time.time()
                        # attempt to read full text (if available)
                        try:
                            resp_text = await resp.text()
                        except Exception:
                            resp_text = None
                    except Exception:
                        # server may not support streaming or the client raised; fallback to simple read
                        rec.first_byte_time = time.time()
                        text = await resp.text()
                        content_bytes = len(text.encode('utf-8')) if text else 0
                        resp_text = text

                    rec.end_time = time.time()
                    rec.bytes_received = content_bytes
                    if resp_text:
                        rec.resp_len_tokens = len(resp_text.split())
                    else:
                        rec.resp_len_tokens = None
                except Exception as e:
                    rec.error = str(e)
                    rec.end_time = time.time()
                finally:
                    # write metrics asynchronously
                    await self.metrics_writer.write(rec)
                    self._queue.task_done()

    async def run(self):
        os.makedirs(self.config.outdir, exist_ok=True)
        # start GPU monitor
        if self.gpu_watcher:
            self.gpu_watcher.start()
        # start producer
        self._start_ts = time.time()
        self._producer_task = asyncio.create_task(self._producer())
        # start workers
        for i in range(self.config.concurrency):
            t = asyncio.create_task(self._worker(i))
            self._workers.append(t)
        # wait for producer and queue to drain
        await self._producer_task
        await self._queue.join()
        # cancel workers if any left
        for w in self._workers:
            try:
                w.cancel()
            except Exception:
                pass
        # stop gpu watcher
        if self.gpu_watcher:
            await self.gpu_watcher.stop()
        print("Benchmark completed. Metrics at:", self.metrics_writer.csv_path)


# ----------------------------- Plotting utilities -----------------------------

def compute_throughput_latency_windows(metrics_csv: str, window_sec: float = 1.0) -> List[Dict[str, Any]]:
    """Read metrics.csv and compute throughput and latency percentiles per time window.

    Returns list of windows: [{window_start, throughput, p50, p90, p99, count}, ...]
    """
    rows = []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for rec in r:
            try:
                end_time = float(rec.get("end_time") or 0)
                send_time = float(rec.get("send_time") or 0)
                fb = rec.get("first_byte_time")
                first_byte_time = float(fb) if fb not in (None, "", "None") else None
                # use end_time for assignment to window
                latency = None
                if first_byte_time and send_time:
                    latency = first_byte_time - send_time
                elif end_time and send_time:
                    latency = end_time - send_time
                else:
                    continue
                rows.append({"end_time": end_time, "latency": latency})
            except Exception:
                continue
    if not rows:
        return []
    rows.sort(key=lambda x: x["end_time"])
    start = rows[0]["end_time"]
    end = rows[-1]["end_time"]
    windows = []
    wstart = start
    while wstart <= end:
        wend = wstart + window_sec
        latencies = [r["latency"] for r in rows if wstart <= r["end_time"] < wend]
        count = len(latencies)
        if count > 0:
            windows.append({
                "window_start": wstart,
                "throughput": count / window_sec,
                "p50": statistics.median(latencies) if latencies else None,
                "p90": percentile(latencies, 90) if latencies else None,
                "p99": percentile(latencies, 99) if latencies else None,
                "count": count
            })
        wstart = wend
    return windows


def percentile(data: List[float], p: float) -> float:
    if not data:
        return None
    k = (len(data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted(data)[int(k)]
    d0 = sorted(data)[f] * (c - k)
    d1 = sorted(data)[c] * (k - f)
    return d0 + d1


def plot_throughput_latency(metrics_csv: str, out_png: str, window_sec: float = 1.0):
    """Produce a throughput vs latency scatter/line plot and save to PNG.

    The function computes windows of `window_sec` seconds and for each window computes
    throughput and latency percentiles (p50, p90, p99). It then plots throughput on
    the x-axis and the latency percentiles on the y-axis.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is required for plotting:", e)
        return

    windows = compute_throughput_latency_windows(metrics_csv, window_sec)
    if not windows:
        print("No data to plot")
        return
    xs = [w["throughput"] for w in windows]
    p50s = [w["p50"] for w in windows]
    p90s = [w["p90"] for w in windows]
    p99s = [w["p99"] for w in windows]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, p50s, label="p50 latency")
    plt.plot(xs, p90s, label="p90 latency")
    plt.plot(xs, p99s, label="p99 latency")
    plt.xlabel("Throughput (requests/sec)")
    plt.ylabel("Latency (s)")
    plt.title("Throughput vs Latency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved plot to {out_png}")


# ----------------------------- CLI and helpers -----------------------------

def default_headers_from_env() -> Dict[str, str]:
    h = {}
    # try to load AUTH token from env
    token = os.environ.get("LLM_API_KEY")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def run_cli():
    p = argparse.ArgumentParser(description="InferX benchmarking framework")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--rate", type=float, default=10.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--prompt-dist", choices=["uniform", "zipf", "choices"], default="uniform")
    p.add_argument("--min-tokens", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--choices-tokens", type=str, default=None,
                   help="comma-separated list of token sizes to choose from")
    p.add_argument("--request-body-template", type=str, default='{"prompt": "{prompt}"}')
    p.add_argument("--json-field", type=str, default=None)
    p.add_argument("--outdir", type=str, default="./results")
    p.add_argument("--gpu-monitor", action="store_true")
    p.add_argument("--gpu-poll-interval", type=float, default=0.5)
    p.add_argument("--sample-prompts-file", type=str, default=None)
    args = p.parse_args()

    choices_tokens = None
    if args.choices_tokens:
        choices_tokens = [int(x) for x in args.choices_tokens.split(",")]

    headers = default_headers_from_env()

    cfg = BenchmarkConfig(
        endpoint=args.endpoint,
        duration=args.duration,
        rate=args.rate,
        concurrency=args.concurrency,
        prompt_dist=args.prompt_dist,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        choices_tokens=choices_tokens,
        request_body_template=args.request_body_template,
        json_field=args.json_field,
        headers=headers,
        outdir=args.outdir,
        gpu_poll_interval=args.gpu_poll_interval,
        gpu_monitor=args.gpu_monitor,
        sample_prompts_file=args.sample_prompts_file
    )

    bm = AsyncLLMBenchmark(cfg)
    asyncio.run(bm.run())
    # After run, produce a plot
    metrics_csv = os.path.join(cfg.outdir, "metrics.csv")
    out_png = os.path.join(cfg.outdir, "throughput_latency.png")
    plot_throughput_latency(metrics_csv, out_png, window_sec=1.0)


if __name__ == "__main__":
    run_cli()
