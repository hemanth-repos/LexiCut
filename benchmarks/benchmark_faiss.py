import os
import sys
import time
import numpy as np

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vector_index import FaissVectorIndex

def linear_scan_lookup(query_vector, store):
    """Manually compute cosine similarity against all stored vectors (O(N) linear scan)."""
    best_similarity = -1.0
    best_id = None
    
    norm_q = np.linalg.norm(query_vector)
    if norm_q == 0:
        return None, 0.0
        
    for query_id, embedding in store.items():
        norm_e = np.linalg.norm(embedding)
        if norm_e == 0:
            continue
        similarity = float(np.dot(query_vector, embedding) / (norm_q * norm_e))
        if similarity > best_similarity:
            best_similarity = similarity
            best_id = query_id
            
    return best_id, best_similarity

def run_benchmarks():
    print("==================================================")
    print("LexiCut Cache Retrieval Benchmarking Suite")
    print("Comparing O(N) Linear Scan vs FAISS FlatIP (IDMap)")
    print("==================================================")
    
    sizes = [100, 1000, 5000, 10000]
    dimension = 384
    num_runs = 100
    
    results = {}
    
    # Generate queries to test with
    np.random.seed(42)
    test_queries = [np.random.randn(dimension).astype(np.float32) for _ in range(num_runs)]
    # Normalize test queries
    test_queries = [q / np.linalg.norm(q) for q in test_queries]
    
    for size in sizes:
        print(f"\nGenerating synthetic dataset of size: {size}...")
        # Generate random vectors
        dataset = {i: np.random.randn(dimension).astype(np.float32) for i in range(size)}
        # Normalize dataset vectors
        dataset = {i: d / np.linalg.norm(d) for i, d in dataset.items()}
        
        # 1. Benchmark Linear Scan
        print(f"Benchmarking Linear Scan ({size} entries)...")
        linear_times = []
        for q in test_queries:
            t_start = time.perf_counter()
            linear_scan_lookup(q, dataset)
            linear_times.append((time.perf_counter() - t_start) * 1000.0) # in ms
            
        avg_linear = np.mean(linear_times)
        p95_linear = np.percentile(linear_times, 95)
        
        # 2. Benchmark FAISS Index
        print(f"Benchmarking FAISS Index ({size} entries)...")
        faiss_index_path = f"benchmark_temp_{size}.index"
        if os.path.exists(faiss_index_path):
            os.remove(faiss_index_path)
            
        faiss_idx = FaissVectorIndex(faiss_index_path)
        # Populate FAISS index
        for query_id, embedding in dataset.items():
            faiss_idx.add_vector(query_id, embedding)
        faiss_idx.save_index()
        
        faiss_times = []
        for q in test_queries:
            t_start = time.perf_counter()
            faiss_idx.search(q, top_k=5)
            faiss_times.append((time.perf_counter() - t_start) * 1000.0) # in ms
            
        avg_faiss = np.mean(faiss_times)
        p95_faiss = np.percentile(faiss_times, 95)
        
        # Clean up index file
        if os.path.exists(faiss_index_path):
            try:
                os.remove(faiss_index_path)
            except Exception:
                pass
            
        results[size] = {
            "linear_avg": avg_linear,
            "linear_p95": p95_linear,
            "faiss_avg": avg_faiss,
            "faiss_p95": p95_faiss
        }
        
    print("\n\n==================================================")
    print("BENCHMARK RESULTS COMPARISON TABLE")
    print("==================================================")
    print(f"{'Entries':<10} | {'Linear Scan (Avg)':<20} | {'FAISS (Avg)':<15} | {'Speedup Factor':<15}")
    print("-" * 68)
    for size in sizes:
        res = results[size]
        speedup = res["linear_avg"] / res["faiss_avg"] if res["faiss_avg"] > 0 else 0.0
        print(f"{size:<10,} | {res['linear_avg']:>16.4f} ms | {res['faiss_avg']:>11.4f} ms | {speedup:>14.1f}x")
    print("==================================================")
    
    print("\nP95 LATENCY COMPARISON TABLE")
    print("==================================================")
    print(f"{'Entries':<10} | {'Linear Scan (P95)':<20} | {'FAISS (P95)':<15} | {'Speedup Factor':<15}")
    print("-" * 68)
    for size in sizes:
        res = results[size]
        speedup_p95 = res["linear_p95"] / res["faiss_p95"] if res["faiss_p95"] > 0 else 0.0
        print(f"{size:<10,} | {res['linear_p95']:>16.4f} ms | {res['faiss_p95']:>11.4f} ms | {speedup_p95:>14.1f}x")
    print("==================================================")
    
    # Save benchmark stats to a temp pickle to import in report builder
    import pickle
    stats_path = os.path.join(os.path.dirname(__file__), "benchmark_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nBenchmark results serialized to '{stats_path}'.")

if __name__ == "__main__":
    run_benchmarks()
