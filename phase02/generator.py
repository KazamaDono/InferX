import asyncio
import time
import numpy as np
import aiohttp
import random
import string
import csv
from typing import List, Dict, Optional
import json
import torch
import pynvml
from typing import Any
from plot import MetricsAnalyzer
import argparse

pynvml.nvmlInit()


class PoissonGenerator:
    def __init__(
        self,
        rate: float = 5.0,
        total_requests: int = 20,
        duration: Optional[float] = None,
        word_lambda: float = 15.0,
        endpoint: str = "http://localhost:8000/v1/completions",
        max_concurrency: int = 5
    ):
        self.rate = rate
        self.total_requests = total_requests
        self.duration = duration
        self.word_lambda = word_lambda
        self.endpoint = endpoint
        self.max_concurrency = max_concurrency

    # ---------------------------
    # Prompt generation
    # ---------------------------
    def poisson_word_count(self, min_words: int = 1) -> int:
        val = np.random.poisson(lam=self.word_lambda)
        return max(min_words, val)

    def make_prompt(self, word_count: int, word_len_mean: float = 5.0) -> str:
        words = []
        for _ in range(word_count):
            word_length = max(1, int(random.gauss(word_len_mean, max(1.0, word_len_mean * 0.25))))
            word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
            words.append(word)
        return ' '.join(words)
    
    def get_gpu_memory(self) -> List[Any]:
        mem_list = []
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_list.append(mem_info.used / (1024**2))  # MB
        return mem_list
    # ---------------------------
    # Async Poisson request generator
    # ---------------------------
    async def generate_requests(self):
        seq = 0
        start_time = time.time()
        while True:
            now = time.time()
            if self.duration and (now - start_time) >= self.duration:
                break
            if self.total_requests and seq >= self.total_requests:
                break

            seq += 1
            word_count = self.poisson_word_count()
            prompt = self.make_prompt(word_count)
            yield {"seq": seq, "ts": time.time(), "prompt": prompt, "word_count": word_count}

            await asyncio.sleep(random.expovariate(self.rate))

    # ---------------------------
    # Send a single streaming request
    # ---------------------------
    async def send_request(self, session: aiohttp.ClientSession, req: dict, sem: asyncio.Semaphore, metrics: list):
        async with sem:
            t0 = time.time()
            first_token_time = None
            output_tokens = 0

            try:
                mem_before = self.get_gpu_memory()
                async with session.post(self.endpoint, json={"prompt": req["prompt"], "stream": True}) as resp:
                    # Read NDJSON line by line
                    while True:
                        line_bytes = await resp.content.readline()
                        if not line_bytes:
                            break  # EOF

                        line = line_bytes.decode("utf-8").strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data:"):
                            line = line[len("data:"):].strip()

                        try:
                            data = json.loads(line)
                            token_text = data.get("choices", [{}])[0].get("text", "")
                            if token_text:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                output_tokens += len(token_text.split())
                        except json.JSONDecodeError:
                            continue

                    total_latency = time.time() - t0
                    ttft = first_token_time - t0 if first_token_time else total_latency
                    total_tokens = req["word_count"] + output_tokens
                    tokens_per_sec = total_tokens / total_latency if total_latency > 0 else 0
                    
                    
                    mem_after = self.get_gpu_memory()
                    peak_memory_mb = max(after - before for before, after in zip(mem_before, mem_after))
                    metrics.append({
                        "seq": req["seq"],
                        "input_tokens": req["word_count"],
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "latency": total_latency,
                        "ttft": ttft,
                        "tpot": total_latency / output_tokens if output_tokens > 0 else 0,
                        "throughput_tps": output_tokens / total_latency if total_latency > 0 else 0,
                        "tokens_per_sec": tokens_per_sec,
                        "status": resp.status,
                        "gpu_memory_mb": peak_memory_mb
                    })

            except Exception as e:
                latency = time.time() - t0
                metrics.append({
                    "seq": req["seq"],
                    "input_tokens": req["word_count"],
                    "output_tokens": 0,
                    "total_tokens": req["word_count"],
                    "latency": latency,
                    "ttft": 0,
                    "tokens_per_sec": 0,
                    "status": "error",
                    "error": str(e)
                })

    # ---------------------------
    # Run workload
    # ---------------------------
    async def run(self):
        metrics = []
        sem = asyncio.Semaphore(self.max_concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = []
            async for req in self.generate_requests():
                task = asyncio.create_task(self.send_request(session, req, sem, metrics))
                tasks.append(task)
            await asyncio.gather(*tasks)
        return metrics

    # ---------------------------
    # CSV export
    # ---------------------------
    @staticmethod
    def write_csv(records: List[Dict], filepath: str) -> str:
        if not records:
            return "No records to write"
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            for row in records:
                writer.writerow(row)
        return f"Metrics saved to {filepath}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run PoissonGenerator benchmark")
    parser.add_argument("--rate", type=float, default=5.0, help="Poisson arrival rate")
    parser.add_argument("--total_requests", type=int, default=20, help="Total number of requests")
    parser.add_argument("--duration", type=float, default=None, help="Max duration in seconds")
    parser.add_argument("--word_lambda", type=float, default=15.0, help="Lambda for Poisson word count")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1/completions", help="API endpoint")
    parser.add_argument("--max_concurrency", type=int, default=5, help="Max concurrent requests")
    return parser.parse_args()

# ---------------------------
# Async main entry point
# ---------------------------
async def main():
    args = parse_args()
    gen = PoissonGenerator(
        rate=args.rate,
        total_requests=args.total_requests,
        duration=args.duration,
        word_lambda=args.word_lambda,
        endpoint=args.endpoint,
        max_concurrency=args.max_concurrency
    )
    metrics = await gen.run()
    for m in metrics:
        print(m)
    
    # Assuming 'metrics' is the list returned from your PoissonGenerator.run()
    analyzer = MetricsAnalyzer(metrics)

    # Plot throughput vs latency
    analyzer.plot_throughput_vs_latency("plot/plot_throughput_vs_latency.png")

    # Plot TTFT distribution
    analyzer.plot_ttft_distribution("plot/ttft_distribution.png")

    # Plot tokens per second
    analyzer.plot_tokens_per_sec("plot/tokens_per_sec.png")

    
    gen.write_csv(metrics, "benchmark/benchmark_results.csv")

if __name__ == "__main__":
    asyncio.run(main())
