# InferX
Inference Accelerator

Currently developing a custom asynchronous, queue-based dynamic batching scheduler for large-language model inference. Benchmarked against HuggingFace Transformers and vLLM on a single RTX 3090 GPU. The project aims to demonstrate how dynamic batching can double throughput and reduce latency by ~40% compared to naïve batching, while also comparing against vLLM’s speed and memory efficiency.
