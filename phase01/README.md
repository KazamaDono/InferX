# Phase 01: vLLM Server Setup and OpenAI Client Benchmarking

This project implements a minimal vLLM server setup with an OpenAI-compatible client for performance benchmarking.

## Overview

The codebase consists of three main components:
- **vLLM Server**: A lightweight inference server using the TinyLlama model
- **OpenAI Client**: A Python client that mimics OpenAI's API interface
- **Benchmarking System**: Performance metrics collection and CSV export functionality

## Files

### `start_server.sh`
Shell script that launches the vLLM OpenAI API server with the following configuration:
- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Port**: 8000
- **GPU Memory Utilization**: 90%

### `client.py`
Main Python client implementing the `OpenAIClient` class with the following features:

#### Core Functionality

- **Hardware Detection**: CUDA availability and GPU information (`check_hardware()`)
- **Request Handling**: Both synchronous (`send_request()`) and streaming (`send_streaming_request()`) completions
- **OpenAI Compatibility**: Uses OpenAI Python SDK with local vLLM endpoint

#### Benchmarking Features
- **Metrics Calculation** (`calculate_metrics()`):
  - **TTFT** (Time To First Token): Latency until first token generation
  - **TPOT** (Time Per Output Token): Average time per token
  - **Total Latency**: End-to-end request completion time
  - **Throughput (TPS)**: Tokens per second
  - **Peak Memory**: GPU memory usage tracking

- **CSV Export** (`write_csv()`): Exports benchmark results to CSV format

#### Test Queries
Includes 10 diverse test queries covering:
- General knowledge questions
- Technical explanations
- Creative content requests
- Programming-related queries

### `pyproject.toml`
Project configuration with dependencies:
- **Core**: PyTorch ecosystem (torch, torchvision, torchaudio)
- **LLM**: vLLM (>=0.10.2), Transformers (>=4.56.2)
- **Client**: OpenAI SDK (>=1.109.0)
- **Visualization**: Matplotlib (>=3.10.6)

### `benchmark_results.csv`
Sample benchmark results showing performance metrics for each test query, including cases where requests failed (0 values indicate failed requests).

## Usage

1. **Start the vLLM server**:
   ```bash
   ./start_server.sh
   ```

2. **Run benchmarking**:
   ```bash
   python client.py
   ```

3. **View results**: Check `benchmark_results.csv` for detailed performance metrics.

## Performance Metrics

The benchmarking system measures:
- **Latency**: Response time characteristics
- **Throughput**: Token generation speed
- **Memory Usage**: GPU memory consumption
- **Reliability**: Success/failure rates across multiple queries

This setup provides a foundation for evaluating LLM inference performance in a controlled environment.