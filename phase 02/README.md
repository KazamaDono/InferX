# Benchmark Framework — Phase 2

### Goal of Phase 2

Abstract out a benchmarking framework + base framework to make it modular, clean, reusable, and fast‑to‑understand.

Key requirements from Phase 2:
- Clean: no redundant code; every unit (function/class/data structure) has a single responsibility.
- Modular: atomic features implemented as independent units.
- Fast understanding: code should be instantly understandable by collaborators.
- Reusable: code for a role X must only depend on X (no overlapping responsibilities).

Deliverables / Features (as implemented in the provided files):
- asyncio‑based Poisson arrival workload generator with variable prompt lengths
- Asynchronous workers using aiohttp (streaming‑aware TTFT measurement)
- Metrics collection: TTFT, TPOT, throughput, request timings, status codes
- GPU memory monitoring (pynvml or nvidia‑smi fallback)
- Standardized CSV (and JSON) outputs: metrics.csv, gpu_memory.csv
- Plotting utilities: throughput vs latency (p50/p90/p99) curves
- CLI for quick experiments and reproducible runs

---
### Repository layout
```bash
phase2-benchmark-framework/
├── inferx_benchmark_framework.py 
├── trials/ 
│ └── run1/ 
│   ├── metrics.csv
│   ├── gpu_memory.csv
│   └── throughput_latency.png
│ └── run2/ 
│   ├── metrics.csv
│   ├── gpu_memory.csv
│   └── throughput_latency.png
│ └── run3/ 
│   ├── metrics.csv
│   ├── gpu_memory.csv
│   └── throughput_latency.png
├── prompts.txt 
└── README.md
```
>Note: gpu_memory.csv and metrics.csv will differ between runs — the framework writes these per run to the --outdir you provide.
---

## Step-by-Step Guide

### 1. Setup Environment
   1. SSH into your Vast.ai instance:
      ```bash
      ssh -L 8888:localhost:8888 user@<instance-ip>
      ```
   2. Environment Setup:
      Inside Jupyter Terminal or VS code terminal:
      ```bash
      # Update
      apt update && apt install -y git python3-venv

      # Create project folder
      mkdir vllm_baseline && cd vllm_baseline

      # Virtual environment
      python3 -m venv vllm_env
      source vllm_env/bin/activate

      # Install dependencies
      pip install --upgrade pip
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
      pip install vllm
      pip install transformers matplotlib
      pip install aiohttp numpy matplotlib pandas pynvml
      ```
      At this point, you have PyTorch, vLLM, and Hugging Face installed with GPU support.

### 2. Run vLLM sever
Start the API server with a small model (TinyLlama):
```bash
     python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9
```
- This will download the model (takes a few minutes).
- You’ll see logs that the API is running on port **8000**.
> Note: Run the above code as a one liner:
```bash
python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8000 --gpu-memory-utilization 0.9
```

### 3. Run the benchmark framework
Your uploaded script is ready to run as a CLI tool. Example:
```bash
python inferx_benchmark_framework.py \
    --endpoint http://localhost:8000/v1/generate \
    --duration 60 \
    --rate 20 \
    --concurrency 8 \
    --prompt-dist uniform \
    --min-tokens 8 --max-tokens 128 \
    --outdir ./results/run1 \
    --gpu-monitor
```
Or as a One liner:
```bash
python inferx_benchmark_framework.py --endpoint http://localhost:8000/v1/generate --duration 60 --rate 20 --concurrency 8 --prompt-dist uniform --min-tokens 8 --max-tokens 128 --outdir ./results/run1 --gpu-monitor
```

Another example:
```bash
python inferx_benchmark_framework.py \
    --endpoint http://localhost:8000/v1/completions \
    --duration 120 --rate 50 --concurrency 16 \
    --request-body-template '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","prompt":"{prompt}","max_tokens":128}' \
    --outdir ./results/run2 --gpu-monitor
```

What this does:
- **Workload generator (2.1)** → Poisson arrivals, variable prompt lengths.
- **Metrics collection (2.2)** → logs TTFT, TPOT, latency, throughput, GPU memory.
- **Visualization (2.3)** → generates a `throughput_latency.png` curve automatically.

---

## What the outputs contain

'metrics.csv' — per‑request data with columns such as:
- request_id, enqueue_time, send_time, first_byte_time, end_time, prompt_tokens, resp_tokens, bytes_received, status, error
- TTFT: time to first response chunk (if server doesn't stream, TTFT = total response time)
- TPOT: time per output token = (end_time - first_byte_time) / n_output_tokens (estimated)

'gpu_memory.csv' — periodic GPU memory and utilization samples with timestamps (pynvml or nvidia-smi fallback)

'throughput_latency.png' — throughput (x) vs latency percentiles (p50/p90/p99) (y) produced by the plotting utilities in the script.

> Reminder: gpu_memory.csv and metrics.csv will be different per run — the script collects live run data.

---

## (Optional) Customizations
- Different prompt sizes:
  ```bash
  --prompt-dist choices --choices-tokens 32,64,128
  ```
- Custom request body:
  ```bash
  --request-body-template '{"input":"{prompt}","max_new_tokens":128}'
  ```
- Use a dataset of prompts:
  ```bash
  --sample-prompts-file prompts.txt
  ```

---

## Phase 2 Completion Checklist
- [x]  **Workload generator** (Poisson arrivals, variable prompt length)
- [x]  **Metrics collection** (TTFT, TPOT, throughput, GPU memory, CSV/JSON output)
- [x]  **Visualization tools** (throughput vs latency curve)
