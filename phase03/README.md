# Phase 03: Naive Scheduler Implementation

This project implements a basic batch scheduler for LLM inference using transformers and explores both synchronous and asynchronous scheduling approaches.

## Overview

Phase 03 demonstrates fundamental batching concepts for LLM inference optimization. The codebase consists of two scheduler implementations:
- **Naive Scheduler**: Synchronous batch-based request processing
- **Async Naive Scheduler**: Asynchronous batch scheduler with timeout-based batching
- **Benchmarking System**: Performance measurement and comparison framework

## Directory Structure

```
phase03/
├── naive_scheduler.py          # Synchronous batch scheduler
├── async_naive_scheduler.py    # Asynchronous batch scheduler
├── run.py                      # Benchmark runner with metrics collection
├── pyproject.toml              # Project dependencies
└── benchmark_results.csv       # Benchmark output (generated)
```

## Core Concepts to Remember

### 1. Batching Fundamentals

**Why Batching?**
- GPUs are optimized for parallel computation
- Processing multiple requests together maximizes GPU utilization
- Batching reduces per-request overhead
- Trade-off: slight latency increase for individual requests vs. much higher overall throughput

**Batching Strategy**:
- Collect requests until batch is full OR timeout expires
- Process entire batch together through the model
- Distribute results back to individual requesters

### 2. Synchronous vs Asynchronous Scheduling

| Aspect | Naive Scheduler | Async Naive Scheduler |
|--------|----------------|----------------------|
| Execution Model | Blocking/Synchronous | Non-blocking/Asynchronous |
| Request Handling | Sequential queue | Concurrent queue (asyncio.Queue) |
| Batch Trigger | Size-based only | Size-based OR timeout |
| Concurrency | None | Multiple concurrent requests |
| Use Case | Simple demo | Production-like scenarios |

## Component Deep Dive

### 1. `naive_scheduler.py` - Synchronous Batch Scheduler

**Architecture**:
```python
class NaiveScheduler:
    - batch_size: int      # Number of requests to batch together
    - queue: List[str]     # Simple list to hold pending prompts
```

**Key Methods**:
- `add_request(prompt)`: Adds prompt to queue, processes batch if full
- `run_batch()`: Tokenizes, generates, and decodes batch of prompts

**Important Implementation Details**:
1. **Padding**: Uses `tokenizer(..., padding=True)` to handle variable-length inputs
2. **Token Extraction**: Only decodes NEW tokens (not input prompt)
   ```python
   input_length = inputs["input_ids"].shape[1]
   new_tokens = outputs[:, input_length:]
   ```
3. **Generation Parameters**:
   - `max_new_tokens=50`: Limits output length
   - `do_sample=True`: Enables stochastic sampling
   - `top_k=50, top_p=0.95, temperature=0.7`: Controls generation diversity

**Limitations**:
- Blocks until batch is full
- Last batch may have fewer items (remainder processing needed)
- No timeout mechanism
- Synchronous execution (no concurrency)

### 2. `async_naive_scheduler.py` - Asynchronous Batch Scheduler

**Architecture**:
```python
class AsyncNaiveScheduler:
    - batch_size: int           # Max requests per batch
    - timeout: float            # Max wait time for batch to fill
    - queue: asyncio.Queue      # Async queue for (prompt, future) pairs
    - running: bool             # Background loop control flag
    - batch_count: int          # Track number of batches processed
```

**Key Async Patterns**:

1. **Future-Based Request Handling**:
   ```python
   async def add_request(prompt):
       fut = asyncio.get_event_loop().create_future()  # Create promise
       await self.queue.put((prompt, fut))              # Enqueue
       return await fut                                 # Wait for result
   ```
   - Creates a Future for each request
   - Request blocks on Future until result is set
   - Decouples request submission from batch processing

2. **Background Batch Loop**:
   ```python
   async def _batch_loop():
       while self.running:
           # Collect batch until size OR timeout
           # Process batch
           # Set results on futures
   ```
   - Runs continuously in background (`asyncio.create_task`)
   - Collects requests with dynamic timeout calculation
   - Processes batch and resolves futures

3. **Dynamic Timeout Calculation**:
   ```python
   timeout_remaining = self.timeout - (loop.time() - start_time)
   item = await asyncio.wait_for(queue.get(), timeout=timeout_remaining)
   ```
   - Tracks elapsed time since batch started
   - Ensures batch processes even if not full after timeout
   - Handles `TimeoutError` gracefully

4. **Graceful Shutdown**:
   ```python
   async def shutdown():
       self.running = False          # Stop loop
       # Process remaining queue items
       self.task.cancel()            # Cancel background task
       await self.task               # Wait for cleanup
   ```

**Critical Implementation Notes**:
- `tokenizer.pad_token = tokenizer.eos_token`: Essential for batch processing
- `model.to(device)`: Move model to GPU before inference
- Error handling: Sets exceptions on futures if batch processing fails
- Queue draining: Processes all remaining requests on shutdown

### 3. `run.py` - Benchmark Framework

**Benchmark Configuration**:
- `n_runs = 3`: Multiple runs for statistical reliability
- `batch_size = 4`: Requests per batch
- `timeout = 0.5`: Max wait time for batch to fill (seconds)
- `max_concurrent = 10`: Semaphore limit on concurrent requests
- `30 prompts`: Diverse test prompts for realistic evaluation

**Metrics Collected**:

| Metric | Description | Calculation |
|--------|-------------|-------------|
| `run_num` | Which benchmark run | 1-based index |
| `seq` | Request sequence number | 1-based index |
| `status` | Success/error status | "ok" or "error" |
| `batch_num` | Batch this request was in | scheduler.batch_count |
| `input_tokens` | Word count of prompt | len(prompt.split()) |
| `output_tokens` | Word count of output | len(output.split()) |
| `total_tokens` | Sum of input + output | - |
| `latency` | Total request time (s) | end_time - start_time |
| `ttft` | Time to first token (s) | Same as latency (naive impl) |
| `tokens_per_sec` | Generation throughput | tokens / latency |
| `prompt` | Original input | - |
| `output_prompt` | Generated text (cleaned) | Whitespace normalized |
| `error` | Error message if failed | Exception string |

**Key Functions**:

1. **`send_request(req, scheduler, metrics, sem)`**:
   - Wraps request sending with timing and metrics collection
   - Uses semaphore to limit concurrency
   - Cleans output (removes newlines/extra whitespace)
   - Appends metrics to shared list

2. **`print_metrics(metrics)`**:
   - Displays summary statistics
   - Shows successful vs error count
   - Calculates averages (latency, tokens/sec)
   - Lists detailed per-request metrics

3. **`write_to_csv(metrics, filepath)`**:
   - Exports to CSV with defined column order
   - Sorts by run_num then seq
   - Uses `DictWriter` with `extrasaction='ignore'`

**Benchmark Execution Flow**:
```
For each run:
  1. Create new AsyncNaiveScheduler instance
  2. Create request objects with prompts
  3. Send all requests concurrently (asyncio.gather)
  4. Shutdown scheduler (process remaining queue)
  5. Print run-specific metrics
  6. Accumulate to all_metrics

After all runs:
  7. Write cumulative CSV
  8. Print summary statistics
```

## Step-by-Step Usage Guide

### Prerequisites

1. **Python 3.11+** environment
2. **Install dependencies**:
   ```bash
   uv sync
   # or
   pip install torch transformers
   ```
3. **GPU (optional)**: Code detects CUDA availability automatically

### Running the Naive Scheduler (Demo)

```bash
python naive_scheduler.py
```

**What happens**:
- Loads GPT-2 model
- Creates scheduler with batch_size=3
- Processes 5 example prompts
- Prints batch results when batch is full
- Processes remaining prompts at end

**Example output**:
```
Batch completed:
Prompt: The future of AI is
Generated: [50 tokens of generated text]

Prompt: Once upon a time
Generated: [50 tokens of generated text]

Prompt: In a galaxy far away
Generated: [50 tokens of generated text]

Processing remaining prompts...
Prompt: Python is great because
Generated: [50 tokens of generated text]

Prompt: Data science is
Generated: [50 tokens of generated text]
```

### Running the Async Benchmark

```bash
python run.py
```

**What happens**:
1. Runs 3 complete benchmark runs
2. Each run:
   - Creates fresh scheduler
   - Sends 30 prompts concurrently (max 10 concurrent)
   - Processes in batches of 4 with 0.5s timeout
   - Collects detailed metrics
3. Saves results to `benchmark_results.csv`
4. Prints summary statistics

**Expected output**:
```
============================================================
STARTING RUN #1/3
============================================================

Configuration:
  Prompts: 30
  Batch size: 4
  Device: cuda
  Max concurrent: 10
  Timeout: 0.5s
------------------------------------------------------------

Run #1 Results:
============================================================
METRICS
============================================================

Total requests: 30
Successful: 30
Errors: 0

Average latency: 0.612s
Average tokens/sec: 27.45
Total batches: 8

[Detailed per-request metrics...]
```

## Key Learnings & Concepts

### 1. Batch Size Selection

**Trade-offs**:
- **Small batches (1-4)**: Lower latency, underutilized GPU
- **Medium batches (4-16)**: Balanced latency/throughput
- **Large batches (16+)**: Higher throughput, increased latency, memory constraints

**Optimal batch size depends on**:
- Model size (memory footprint)
- Available GPU memory
- Latency requirements
- Request arrival rate

### 2. Timeout Configuration

**Purpose**: Prevent indefinite waiting when requests arrive slowly

**Selection criteria**:
- Too short: Creates small batches (inefficient)
- Too long: Increases individual request latency
- Typical values: 50ms - 500ms

**Formula**: `timeout = target_latency - inference_time`

### 3. Async Programming Benefits

**Why asyncio?**:
- Handle multiple requests without threads
- Event loop efficiently manages I/O-bound operations
- Futures/promises pattern for clean async coordination
- Built-in queue, semaphore, timeout primitives

**Async primitives used**:
- `asyncio.Queue`: Thread-safe async queue
- `asyncio.Future`: Promise-like result container
- `asyncio.Semaphore`: Concurrency limiter
- `asyncio.wait_for`: Timeout wrapper
- `asyncio.gather`: Parallel task execution
- `asyncio.create_task`: Background task creation

### 4. Token Handling

**Padding**:
- Required for batching variable-length sequences
- `tokenizer.pad_token = tokenizer.eos_token`
- `padding=True` in tokenizer call
- Attention masks automatically handled

**Output Extraction**:
```python
input_length = inputs["input_ids"].shape[1]  # Get input seq length
new_tokens = outputs[:, input_length:]       # Slice only new tokens
```
- Why: `model.generate()` returns input + generated tokens
- We only want to show the generated portion

### 5. Generation Parameters

| Parameter | Purpose | Effect |
|-----------|---------|--------|
| `max_new_tokens` | Limit output length | Controls memory & latency |
| `do_sample=True` | Enable randomness | More diverse outputs |
| `temperature` | Control randomness | Higher = more random |
| `top_k` | Limit vocabulary | Sample from top K tokens |
| `top_p` | Nucleus sampling | Sample from cumulative prob p |

**Recommended defaults**:
- Chat/creative: `temperature=0.7-0.9, top_p=0.9-0.95`
- Code/factual: `temperature=0.1-0.3, top_p=0.9`

### 6. Performance Metrics

**Latency (per-request)**:
- Individual user experience metric
- Increases with batch size
- Target: < 1-2s for interactive applications

**Throughput (system-wide)**:
- System capacity metric (requests/sec or tokens/sec)
- Increases with batch size
- Target: Maximize subject to latency constraints

**Batch efficiency**:
- Monitor `batch_num` in results
- Fewer batches = better batching efficiency
- Expect ~7-8 batches for 30 requests with batch_size=4

## Common Patterns & Best Practices

### 1. Scheduler Initialization
```python
scheduler = AsyncNaiveScheduler(
    modelname="gpt2",      # Or any HuggingFace model
    batch_size=4,          # Balance latency/throughput
    device="cuda",         # Auto-detect: cuda if available
    timeout=0.5            # 500ms max wait
)
```

### 2. Sending Requests
```python
async def process_prompt(prompt):
    result = await scheduler.add_request(prompt)
    return result

# Concurrent processing
results = await asyncio.gather(*[
    process_prompt(p) for p in prompts
])
```

### 3. Graceful Cleanup
```python
try:
    # Process requests
    await scheduler.add_request(prompt)
finally:
    # Always shutdown to process remaining queue
    await scheduler.shutdown()
```

### 4. Error Handling
```python
try:
    output = await scheduler.add_request(prompt)
except Exception as e:
    # Future will contain exception from batch processing
    logging.error(f"Request failed: {e}")
```

## Limitations & Future Improvements

**Current Limitations**:
1. **No continuous batching**: Batches are discrete (vs. iteration-level batching)
2. **Fixed max_new_tokens**: All requests generate same number of tokens
3. **No prioritization**: All requests treated equally
4. **Simple scheduling**: No advanced algorithms (e.g., shortest-job-first)
5. **Memory unbounded**: Queue can grow indefinitely
6. **TTFT = latency**: No streaming support (true TTFT measurement)

**Potential Enhancements**:
1. **Continuous batching**: Add/remove requests mid-generation
2. **Dynamic batching**: Adjust batch size based on load
3. **Priority queues**: Support request prioritization
4. **Streaming responses**: True TTFT measurement
5. **Memory limits**: Backpressure when queue too large
6. **Load balancing**: Multiple model replicas
7. **Request cancellation**: Timeout/cancel in-flight requests

## Dependencies

From `pyproject.toml`:
- **transformers** (>=4.56.2): Model loading and inference
- **torch, torchvision, torchaudio**: PyTorch ecosystem
- **vllm** (>=0.10.2): Advanced inference engine (not used in this phase)
- **openai** (>=1.109.0): OpenAI API client (for future use)
- **nvidia-ml-py3** (>=7.352.0): GPU monitoring
- **matplotlib** (>=3.10.6): Plotting (not used in this phase)

## Troubleshooting

**Issue**: `RuntimeError: CUDA out of memory`
- **Solution**: Reduce `batch_size` or use smaller model

**Issue**: Requests timing out
- **Solution**: Increase `timeout` parameter

**Issue**: Low throughput
- **Solution**: Increase `batch_size` and `max_concurrent`

**Issue**: High latency
- **Solution**: Decrease `batch_size` or `timeout`

**Issue**: `pad_token` warnings
- **Solution**: Explicitly set `tokenizer.pad_token = tokenizer.eos_token`

## Next Steps

After understanding Phase 03, typical progression:
1. **Phase 04**: Implement continuous batching (iteration-level scheduling)
2. **Phase 05**: Add PagedAttention for memory efficiency
3. **Phase 06**: Multi-GPU support and model parallelism
4. **Phase 07**: Production deployment with monitoring

## Summary

**Key Takeaways**:
- Batching is essential for GPU utilization in LLM serving
- Async programming enables efficient concurrent request handling
- Trade-off between latency (per-request) and throughput (system-wide)
- Timeout-based batching prevents indefinite waiting
- Futures/promises pattern cleanly separates request submission from processing
- Proper benchmarking requires multiple runs and comprehensive metrics

**This phase demonstrates**: The fundamental building blocks of LLM inference serving, providing a foundation for understanding more advanced systems like vLLM, TensorRT-LLM, and commercial offerings.
