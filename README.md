# InferX
Inference Accelerator

Currently developing a custom asynchronous, queue-based dynamic batching scheduler for large-language model inference. Benchmarked against HuggingFace Transformers and vLLM on a single RTX 3090 GPU. The project aims to demonstrate how dynamic batching can double throughput and reduce latency by ~40% compared to naïve batching, while also comparing against vLLM’s speed and memory efficiency.
<img width="1536" height="1024" alt="inferX" src="https://github.com/user-attachments/assets/313c3649-5719-4e78-8658-6b0c6b9184fb" />
