# vLLM Baseline — Phase 1

### Repository layout
```bash
phase 01/
├─ README.md
├─ requirements.txt
├─ client.py
├─ results_sample.csv
├─ Phase_1._vllm_baseline_setup.pdf
└─ screenshots/ # add screenshots here
```
---

# Phase 1. vLLM Baseline Setup


**Result statement:** “I can run vLLM with a small model, send requests, and log basic metrics (TTFT, TPOT, latency, throughput, memory).”


**Goal:** Establish a working vLLM server and benchmark simple workloads to build familiarity.


**Time:** 1 day


---


## 1.1 Environment setup


Install PyTorch, vLLM, Hugging Face Transformers, matplotlib.


Verify CUDA and GPU drivers.


Pre-setup


Environment setup (Phase 1.1)
Inside Jupyter or VS Code terminal:


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
```
At this point, you have PyTorch, vLLM, and Hugging Face installed with GPU support.


## 1.2 Run vLLM server with a small model

Start the API server with a small model (TinyLlama):

```bash
python -m vllm.entrypoints.openai.api_server \
--model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
--port 8000 \
--gpu-memory-utilization 0.9
```

This will download the model (takes a few minutes).

Note: Run the above code as a one liner:
```bash
python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8000 --gpu-memory-utilization 0.9
```
You’ll see logs that the API is running on port 8000.

## 1.3 Minimal client script
Create a file client.py that sends requests via OpenAI-compatible API and measures TTFT, TPOT, total latency, throughput, peak GPU memory. Save results to CSV.

The provided client.py is included in this repo.

Example usage (run in another terminal while the server is running):
```bash
python client.py
```
> Run the client on Vast.ai (not Windows)
> FYI - we are using a 1x RTX 3090 GPU based on CUDA
Something like in the following (i.e. change the area based closest to your area.
<img width="808" height="110" alt="image" src="https://github.com/user-attachments/assets/e4b8ee2f-f360-4932-afcf-91feb7ba4616" />

---

## 1.4 Guide

Here’s what to do:

1. Open a new terminal in Vast.ai (second console window).

2. In Vast web UI → go to your instance → “Open Console” → you’ll get a new shell.

3. Create the client file inside Vast:
   ```bash
   nano client.py
   ```
    <img width="1255" height="1266" alt="image" src="https://github.com/user-attachments/assets/d944f2e6-e810-4e71-83b6-cebd7a5aa4a1" />
    Save with CTRL+O, Enter, CTRL+X.

4. Run it inside Vast:
   ```bash
   python client.py
    ```
   Now it will connect successfully, since localhost:8000 exists inside Vast.
   <img width="1274" height="285" alt="1234" src="https://github.com/user-attachments/assets/e2ddf023-6b05-4dff-81c8-b0aa6f220f7b" />


5. To check GPU memory usage → type in the Vast.ai console:
   ```bash
    nvidia-smi
    ```
    This will show GPU utilization & VRAM usage for your model.
    <img width="1275" height="685" alt="12" src="https://github.com/user-attachments/assets/6f268c93-db4a-4cc7-a073-accd2b407ffe" />

---
## Pro-Tip
> Shutdown the machine afterwards
When done, go back to Vast and click Stop / Destroy Instance — otherwise it keeps billing.

At this point, you’ll have:

- vLLM server running on GPU
- Client sending requests
- Logged metrics in `results.csv`
- GPU memory usage from `nvidia-smi`

That fully completes **Phase 1**.


