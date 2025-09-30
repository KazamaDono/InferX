# Phase 02: Advanced Workload Generator and Performance Analysis

This project implements an asyncio-based workload generator with Poisson arrivals for benchmarking LLM inference servers, along with comprehensive metrics collection and visualization tools.

## Overview

Phase 02 builds a reusable framework for load testing and performance analysis of LLM inference endpoints. The codebase consists of three main components:
- **Poisson Workload Generator**: Simulates realistic request patterns with configurable arrival rates
- **Metrics Collection System**: Captures detailed performance metrics (TTFT, TPOT, throughput, GPU memory)
- **Visualization Tools**: Generates publication-quality plots for performance analysis

## Directory Structure

```
phase02/
├── generator.py           # Main workload generator with metrics collection
├── plot.py               # Visualization and analysis tools
├── client.py             # Simple requirements documentation
├── pyproject.toml        # Project dependencies
├── benchmark/            # Output directory for CSV results
│   └── benchmark_results.csv
└── plot/                 # Output directory for visualizations
    ├── plot_throughput_vs_latency.png
    ├── ttft_distribution.png
    └── tokens_per_sec.png
```

## Core Components

### 1. `generator.py` - Workload Generator

The main script implementing the `PoissonGenerator` class with comprehensive benchmarking capabilities.

#### Key Features

**Request Generation**:
- **Poisson Arrivals**: Simulates realistic traffic patterns using exponential inter-arrival times
- **Variable Prompt Lengths**: Generates prompts with Poisson-distributed word counts (configurable λ)
- **Concurrency Control**: Configurable max concurrent requests using semaphores

**Metrics Collection**:
- **TTFT (Time To First Token)**: Measures initial response latency
- **TPOT (Time Per Output Token)**: Average generation time per token
- **Total Latency**: End-to-end request completion time
- **Throughput (TPS)**: Tokens generated per second
- **GPU Memory Tracking**: Monitors GPU memory usage before/after requests using NVML

**Configuration Parameters** (via command-line arguments):
- `--rate`: Poisson arrival rate (requests/sec, default: 5.0)
- `--total_requests`: Total number of requests to generate (default: 20)
- `--duration`: Optional max duration in seconds
- `--word_lambda`: Lambda for Poisson word count distribution (default: 15.0)
- `--endpoint`: API endpoint URL (default: http://localhost:8000/v1/completions)
- `--max_concurrency`: Maximum concurrent requests (default: 5)

#### How It Works

1. **Prompt Generation** (`make_prompt()`):
   - Determines word count using Poisson distribution
   - Generates random words with Gaussian-distributed lengths
   - Creates realistic variable-length prompts

2. **Request Generation** (`generate_requests()`):
   - Async generator yielding requests at Poisson-distributed intervals
   - Continues until reaching `total_requests` or `duration` limit
   - Each request includes sequence number, timestamp, prompt, and word count

3. **Request Sending** (`send_request()`):
   - Sends streaming POST requests to the LLM endpoint
   - Parses NDJSON streaming responses
   - Tracks timing for TTFT and total latency
   - Monitors GPU memory usage
   - Collects comprehensive metrics for each request

4. **Workload Execution** (`run()`):
   - Creates async tasks for all requests
   - Manages concurrency with semaphores
   - Returns complete metrics list

5. **Data Export** (`write_csv()`):
   - Exports metrics to CSV format for further analysis

### 2. `plot.py` - Visualization Tools

Implements the `MetricsAnalyzer` class for generating performance visualizations.

#### Visualization Methods

**Throughput vs Latency Plot** (`plot_throughput_vs_latency()`):
- Scatter plot of all measurement points
- **Pareto Frontier**: Highlights optimal performance points
- Performance curve showing best achievable trade-offs
- Ideal target marker showing best latency and throughput
- Visual guidance toward better performance region

**TTFT Distribution** (`plot_ttft_distribution()`):
- Histogram showing distribution of time-to-first-token
- Helps identify latency consistency and outliers

**Tokens Per Second** (`plot_tokens_per_sec()`):
- Line plot showing throughput over request sequence
- Reveals performance trends and potential degradation over time

### 3. `client.py`

Contains requirements documentation for the phase 02 implementation:
- Asyncio-based request generator with Poisson arrivals
- Variable prompt length support
- Comprehensive metrics collection (TTFT, TPOT, throughput, GPU memory)
- Standardized CSV/JSON storage
- Plotting utilities for performance curves

### 4. `pyproject.toml`

Defines project dependencies:
- **PyTorch ecosystem**: torch, torchvision, torchaudio
- **LLM inference**: vLLM (>=0.10.2), transformers (>=4.56.2)
- **HTTP client**: aiohttp (async requests)
- **GPU monitoring**: nvidia-ml-py3 (>=7.352.0)
- **Visualization**: matplotlib (>=3.10.6)
- **Data processing**: numpy

## Step-by-Step Usage Guide

### Prerequisites

1. Ensure vLLM server is running (e.g., from phase01 or standalone)
2. Python 3.11+ environment with dependencies installed

### Basic Workflow

**Step 1: Run the workload generator**
```bash
python generator.py
```

This will:
- Generate 20 requests with Poisson arrivals (rate=5.0 req/sec)
- Send requests to http://localhost:8000/v1/completions
- Collect performance metrics for each request
- Print metrics to console

**Step 2: View generated outputs**
- **CSV Results**: `benchmark/benchmark_results.csv`
- **Visualizations**:
  - `plot/plot_throughput_vs_latency.png` - Performance frontier analysis
  - `plot/ttft_distribution.png` - Response time distribution
  - `plot/tokens_per_sec.png` - Throughput over time

### Advanced Usage

**Custom arrival rate (10 requests/sec)**:
```bash
python generator.py --rate 10.0
```

**Longer stress test (100 requests)**:
```bash
python generator.py --total_requests 100 --max_concurrency 10
```

**Time-bounded test (run for 60 seconds)**:
```bash
python generator.py --duration 60 --rate 5.0
```

**Custom endpoint and prompt length**:
```bash
python generator.py --endpoint http://192.168.1.100:8000/v1/completions --word_lambda 30.0
```

**Full parameter customization**:
```bash
python generator.py \
  --rate 8.0 \
  --total_requests 50 \
  --word_lambda 20.0 \
  --max_concurrency 8 \
  --endpoint http://localhost:8000/v1/completions
```

## Performance Metrics Explained

The benchmarking system captures and analyzes:

- **TTFT (Time To First Token)**: Critical for perceived responsiveness
- **TPOT (Time Per Output Token)**: Indicates generation efficiency
- **Total Latency**: End-to-end user experience
- **Throughput (TPS)**: Overall system capacity (tokens/sec)
- **GPU Memory**: Resource utilization per request
- **Pareto Frontier**: Optimal performance configurations

## Output Files

### `benchmark/benchmark_results.csv`
CSV file with columns:
- `seq`: Request sequence number
- `input_tokens`: Number of input words
- `output_tokens`: Number of generated words
- `total_tokens`: Sum of input and output
- `latency`: Total request time (seconds)
- `ttft`: Time to first token (seconds)
- `tpot`: Time per output token (seconds)
- `throughput_tps`: Output tokens per second
- `tokens_per_sec`: Total tokens per second
- `status`: HTTP status code
- `gpu_memory_mb`: Peak GPU memory usage (MB)

### Visualization Plots

All plots are saved as high-resolution PNG files in the `plot/` directory for use in reports, presentations, or further analysis.

## Use Cases

1. **Load Testing**: Simulate realistic user traffic patterns
2. **Performance Tuning**: Identify optimal configuration parameters
3. **Capacity Planning**: Determine system limits and scaling requirements
4. **Regression Testing**: Compare performance across model versions or configurations
5. **Research**: Generate data for throughput-latency trade-off analysis

## Expected Development Time

As noted in requirements: **2-3 days** to establish this reusable framework for comprehensive LLM inference benchmarking.