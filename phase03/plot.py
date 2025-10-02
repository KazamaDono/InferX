import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict


class MetricsAnalyzer:
    def __init__(self, metrics: List[Dict]):
        self.metrics = metrics
        
    def _ensure_dir(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def plot_throughput_vs_latency(self, filepath: str):
        """Plot throughput vs latency showing the Pareto frontier (best performance)."""
        self._ensure_dir(filepath)
        
        latencies = np.array([m["latency"] for m in self.metrics])
        throughputs = np.array([m["throughput_tps"] for m in self.metrics])
        
        plt.figure(figsize=(8, 6))
        
        # Scatter all points
        plt.scatter(latencies, throughputs, s=100, color='lightblue', 
                   alpha=0.6, edgecolors='black', linewidth=1, 
                   label='All Measurements', zorder=3)
        
        # Find Pareto frontier: points with lowest latency for their throughput level
        # Or highest throughput for their latency level
        pareto_points = []
        sorted_by_lat = sorted(zip(latencies, throughputs))
        
        max_throughput_seen = 0
        for lat, thr in sorted_by_lat:
            if thr >= max_throughput_seen:
                pareto_points.append((lat, thr))
                max_throughput_seen = thr
        
        if len(pareto_points) > 0:
            pareto_lat, pareto_thr = zip(*pareto_points)
            
            # Highlight Pareto optimal points
            plt.scatter(pareto_lat, pareto_thr, s=150, color='green', 
                       edgecolors='darkgreen', linewidth=2, 
                       label='Pareto Optimal', zorder=5, marker='*')
            
            # Draw curve through Pareto points
            plt.plot(pareto_lat, pareto_thr, '--', linewidth=2.5, 
                    color='red', alpha=0.7, label='Performance Frontier')
        
        # Add ideal region annotation
        best_lat = latencies.min()
        best_thr = throughputs.max()
        plt.scatter([best_lat], [best_thr], s=300, color='gold', 
                   marker='*', edgecolors='orange', linewidth=2,
                   label='Ideal Target', zorder=6)
        
        plt.title("Throughput vs Latency - Performance Frontier")
        plt.xlabel("Latency (s) - Lower is Better")
        plt.ylabel("Throughput (tokens/sec) - Higher is Better")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add arrow pointing to ideal direction
        ax = plt.gca()
        ax.annotate('Better Performance', 
                   xy=(best_lat, best_thr), 
                   xytext=(best_lat + 0.02, best_thr - 10),
                   arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                   fontsize=10, color='orange', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    
    def plot_ttft_distribution(self, filepath: str):
        """Plot TTFT distribution - simple histogram."""
        self._ensure_dir(filepath)
        
        ttft_values = [m["ttft"] for m in self.metrics]
        
        plt.figure(figsize=(8, 6))
        
        plt.hist(ttft_values, bins=15, color='tab:orange', edgecolor='black')
        
        plt.title("TTFT Distribution")
        plt.xlabel("Time to First Token (s)")
        plt.ylabel("Count")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    
    def plot_tokens_per_sec(self, filepath: str):
        """Plot tokens per second over requests - simple line plot."""
        self._ensure_dir(filepath)
        
        tps_values = [m["tokens_per_sec"] for m in self.metrics]
        
        plt.figure(figsize=(8, 6))
        
        plt.plot(range(1, len(tps_values)+1), tps_values, 'o-', 
                linewidth=2, markersize=6, color='tab:green')
        
        plt.title("Tokens per Second over Requests")
        plt.xlabel("Request Sequence")
        plt.ylabel("Tokens/sec")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

