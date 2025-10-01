import time
import csv
import requests

API_URL = "http://localhost:8000/v1/completions"

prompt = "Write a short story about an AI and a human working together."

def benchmark(n_requests=5):
    results = []
    for i in range(n_requests):
        start = time.time()
        response = requests.post(API_URL, json={
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt": prompt,
            "max_tokens": 50,
        })
        ttft = time.time() - start
        text = response.json()["choices"][0]["text"]
        end = time.time()
        latency = end - start
        results.append([i+1, ttft, latency, len(text.split())])
        print(f"Request {i+1}: TTFT={ttft:.3f}s, Latency={latency:.3f}s")
    return results

# Run benchmark
rows = benchmark()

# Save results
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Request", "TTFT", "Latency", "OutputTokens"])
    writer.writerows(rows)

print("Results saved to results.csv")

