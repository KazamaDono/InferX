from async_naive_scheduler import AsyncNaiveScheduler
import asyncio
import torch
import csv
from typing import Dict, Any, List


def print_metrics(metrics: List[Dict[str, Any]]) -> None:
    """Display performance metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    
    successful = [m for m in metrics if m["status"] == "ok"]
    errors = [m for m in metrics if m["status"] == "error"]
    
    print(f"\nTotal requests: {len(metrics)}")
    print(f"Successful: {len(successful)}")
    print(f"Errors: {len(errors)}")
    
    if successful:
        avg_latency = sum(m["latency"] for m in successful) / len(successful)
        avg_tokens_per_sec = sum(m["tokens_per_sec"] for m in successful) / len(successful)
        total_batches = max(m["batch_num"] for m in successful)
        
        print(f"\nAverage latency: {avg_latency:.3f}s")
        print(f"Average tokens/sec: {avg_tokens_per_sec:.2f}")
        print(f"Total batches: {total_batches}")
    
    print("\nDetailed metrics:")
    for m in sorted(metrics, key=lambda x: x["seq"]):
        if m["status"] == "ok":
            print(f"  Seq {m['seq']}: {m['output_tokens']} tokens, "
                  f"{m['latency']:.3f}s, {m['tokens_per_sec']:.2f} tok/s, "
                  f"batch #{m['batch_num']}")
        else:
            print(f"  Seq {m['seq']}: ERROR - {m['error']}")
    
    print("=" * 60)


def write_to_csv(metrics: List[Dict[str, Any]], filepath: str) -> None:
    """
    Write metrics to a CSV file with proper column ordering.
    
    Args:
        metrics: List of metric dictionaries
        filepath: Path to output CSV file
    """
    if not metrics:
        print("No metrics to write!")
        return
    
    # Define logical column order
    fieldnames = [
        "run_num",
        "seq",
        "status",
        "batch_num",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency",
        "ttft",
        "tokens_per_sec",
        "prompt",
        "output_prompt",
        "error"
    ]
    
    # Sort by run number, then sequence number
    sorted_metrics = sorted(
        metrics, 
        key=lambda x: (x.get("run_num", 0), x.get("seq", 0))
    )
    
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
        
        # Write header
        writer.writeheader()
        
        # Write rows
        for metric in sorted_metrics:
            writer.writerow(metric)
    
    print(f"Successfully wrote {len(metrics)} records to {filepath}")


async def send_request(
    req: Dict[str, Any],
    scheduler: AsyncNaiveScheduler,
    metrics: List[Dict[str, Any]],
    sem: asyncio.Semaphore
) -> None:
    """
    Send a single request to the scheduler and track metrics.
    
    Args:
        req: Request dictionary with 'run_num', 'seq', 'prompt', 'word_count'
        scheduler: The AsyncNaiveScheduler instance
        metrics: Shared list to collect performance metrics
        sem: Semaphore to limit concurrent requests
    """
    async with sem:
        # Record start time
        t0 = asyncio.get_event_loop().time()
        
        try:
            # Send request and wait for result
            output = await scheduler.add_request(req["prompt"])
            
            # Clean output: remove newlines and extra whitespace
            cleaned_output = " ".join(output.split())
            
            # Calculate metrics
            total_latency = asyncio.get_event_loop().time() - t0
            ttft = total_latency  # Time to first token (for naive impl, same as total)
            tokens = len(output.split())  # Rough token count
            tokens_per_sec = tokens / total_latency if total_latency > 0 else 0
            batch_num = scheduler.batch_count
            
            # Record successful metrics
            metrics.append({
                "run_num": req["run_num"],
                "seq": req["seq"],
                "status": "ok",
                "batch_num": batch_num,
                "input_tokens": req["word_count"],
                "output_tokens": tokens,
                "total_tokens": req["word_count"] + tokens,
                "latency": total_latency,
                "ttft": ttft,
                "tokens_per_sec": tokens_per_sec,
                "prompt": req["prompt"],
                "output_prompt": cleaned_output,
            })
        except Exception as e:
            # Record error
            metrics.append({
                "run_num": req["run_num"],
                "seq": req["seq"],
                "status": "error",
                "prompt": req.get("prompt", ""),
                "error": str(e)
            })


async def main() -> None:
    """Main function to run the benchmark."""
    
    # 30 diverse prompts
    prompts = [
        "The future of AI is",
        "Once upon a time",
        "In a galaxy far away",
        "Python is great because",
        "Data science is",
        "Machine learning helps us",
        "Deep neural networks can",
        "The weather today is",
        "Climate change affects",
        "Renewable energy sources include",
        "The history of computers began",
        "Quantum computing will",
        "Space exploration has",
        "The human brain is",
        "Artificial intelligence can be used to",
        "The internet changed society by",
        "Virtual reality technology allows",
        "Cybersecurity is important because",
        "Cloud computing enables",
        "Natural language processing helps",
        "Blockchain technology can",
        "Self-driving cars will",
        "Biotechnology advances have",
        "The scientific method involves",
        "Education in the future will",
        "Social media platforms have",
        "Environmental conservation requires",
        "Economic systems are based on",
        "Democracy depends on",
        "Healthcare innovations include"
    ]
    
    # Configuration
    n_runs = 3
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timeout = 0.5
    max_concurrent = 10
    
    # Store metrics from all runs
    all_metrics = []
    
    for r in range(n_runs):
        print(f"\n{'='*60}")
        print(f"STARTING RUN #{r + 1}/{n_runs}")
        print(f"{'='*60}\n")
        
        # Initialize NEW scheduler for each run
        scheduler = AsyncNaiveScheduler(
            modelname="gpt2",
            batch_size=batch_size,
            device=device,
            timeout=timeout
        )
        
        # NEW metrics list for this run only
        run_metrics = []
        
        # Semaphore to limit concurrent requests
        sem = asyncio.Semaphore(max_concurrent)
        
        # Create request objects for this run
        requests = []
        for i, prompt in enumerate(prompts):
            request = {
                "run_num": r + 1,
                "seq": i + 1,
                "prompt": prompt,
                "word_count": len(prompt.split())
            }
            requests.append(request)
        
        print(f"Configuration:")
        print(f"  Prompts: {len(requests)}")
        print(f"  Batch size: {scheduler.batch_size}")
        print(f"  Device: {scheduler.device}")
        print(f"  Max concurrent: {max_concurrent}")
        print(f"  Timeout: {timeout}s")
        print("-" * 60)
        
        # Send all requests concurrently
        tasks = [send_request(req, scheduler, run_metrics, sem) for req in requests]
        await asyncio.gather(*tasks)
        
        # Shutdown scheduler after this run
        await scheduler.shutdown()
        
        # Display metrics for THIS run only
        print(f"\nRun #{r + 1} Results:")
        print_metrics(run_metrics)
        
        # Add to cumulative metrics
        all_metrics.extend(run_metrics)
    
    # Write all runs to CSV
    print(f"\n{'='*60}")
    print("WRITING ALL RUNS TO CSV")
    print(f"{'='*60}")
    write_to_csv(all_metrics, "benchmark_results.csv")
    
    # Summary statistics across all runs
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL RUNS")
    print(f"{'='*60}")
    print(f"Total requests: {len(all_metrics)}")
    
    successful = [m for m in all_metrics if m["status"] == "ok"]
    if successful:
        avg_latency = sum(m["latency"] for m in successful) / len(successful)
        avg_tokens_per_sec = sum(m["tokens_per_sec"] for m in successful) / len(successful)
        
        print(f"Successful: {len(successful)}")
        print(f"Overall average latency: {avg_latency:.3f}s")
        print(f"Overall average tokens/sec: {avg_tokens_per_sec:.2f}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())