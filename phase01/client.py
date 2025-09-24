import torch
from openai import OpenAI
import os
import csv
import time
from typing import Any, List, Dict


class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",  # VLLM endpoint
            api_key="not-needed"
        )
    
    def check_hardware(self):
        """Display CUDA hardware information."""
        print(f"CUDA is available: {torch.cuda.is_available()}")
        print(f"HOW MANY CUDA: {torch.cuda.device_count()}")
        print(f"GPU DEVICE NAME: {torch.cuda.get_device_name(0)}")
    
    def send_request(self, query: str) -> str:
        """Send completion request to the model."""
        resp = self.client.completions.create(
            model=os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),  # Fixed typo
            prompt=query,
            max_tokens=os.environ.get("MAX_TOKENS", 50)  # Added missing comma
            # stream=True
        )
        return resp.choices[0].text
    
    def send_streaming_request(self, query: str) -> Any:
        """Send completion request to the model."""
        resp = self.client.completions.create(
            model=os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),  # Fixed typo
            prompt=query,
            max_tokens=os.environ.get("MAX_TOKENS", 50),  # Added missing comma
            stream=True
        )
        
        return resp  
      
    def calculate_metrics(self, queries: List[str]) -> Dict[str, Any]:
        # Initialize lists
        ttft_list = []
        tpot_list = []
        total_latency_list = []
        throughput_tps_list = []
        peak_memory_list = []

        # global_token_count = 0
        # global_start_time = time.time()

        for query in queries:
            # Per-query timing (your existing code is correct here)
            start_time = time.time()
            first_token_time = None
            token_count = 0

            response = self.send_request(query)
            for event in response:
                if first_token_time is None:
                    first_token_time = time.time()
                token_count += 1
                # global_token_count += 1

            end_time = time.time()

            # Per-query metrics (correct)
            ttft = first_token_time - start_time if first_token_time else 0
            tpot = (end_time - start_time) / token_count if token_count > 0 else 0
            total_latency = end_time - start_time
            throughput_tps = token_count / total_latency if total_latency > 0 else 0
            
            # Peak GPU memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                peak_memory_mb = 0

            # Append metrics
            ttft_list.append(ttft)
            tpot_list.append(tpot)
            total_latency_list.append(total_latency)
            throughput_tps_list.append(throughput_tps)
            peak_memory_list.append(peak_memory_mb)

        # # CORRECTED: Global metrics calculation
        # global_end_time = time.time()
        # global_latency = global_end_time - global_start_time
        
        # # Global throughput calculations
        # throughput_rps_global = len(queries) / global_latency if global_latency > 0 else 0
        # throughput_tps_global = global_token_count / global_latency if global_latency > 0 else 0
        
        # # Average metrics (additional useful metrics)
        # avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
        # avg_latency = sum(total_latency_list) / len(total_latency_list) if total_latency_list else 0
        
        # Build result dictionary
        metrics_dict = {
            "ttft": ttft_list,
            "tpot": tpot_list,
            "total_latency": total_latency_list,
            "throughput_tps": throughput_tps_list,
            "peak_memory_mb": peak_memory_list,
        }

        return metrics_dict

    def write_csv(self, records: Dict, filepath: str) -> str:
        """Write dictionary with lists to CSV file."""
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header
            writer.writerow(records.keys())
            
            # Write data rows (fixed: was writing single row)
            if records:
                num_rows = len(list(records.values())[0])  # Get length of first list
                for i in range(num_rows):
                    row = [records[key][i] for key in records.keys()]
                    writer.writerow(row)
        
        return f"Successfully wrote file to {filepath}"


if __name__ == "__main__":
    openai_client = OpenAIClient()
    
    # Check hardware
    # openai_client.check_hardware()
    
    # # Test single request
    # response = openai_client.send_request("Who is the US President?")
    # print(f"Response from the server: {response}")
    # print("\n\n")
    
    # Calculate metrics for multiple queries
    test_queries = [
    "Who is the US President?",
    "What is machine learning?",
    "Explain neural networks briefly.",
    "Write a short poem about technology.",
    "How do computers process information?",
    "What are the benefits of renewable energy?",
    "Describe the concept of artificial intelligence.",
    "Tell me a joke about programming.",
    "What is the difference between Python and JavaScript?",
    "Explain quantum computing in simple terms."
    ]
    
    metrics = openai_client.calculate_metrics(test_queries)
    result = openai_client.write_csv(metrics, "benchmark_results.csv")
    print(result)