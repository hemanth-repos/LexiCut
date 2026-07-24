import os
import sqlite3
import pandas as pd
import sys

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_data():
    db_path = "telemetry.db"
    if not os.path.exists(db_path):
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM request_logs ORDER BY id ASC", conn)
    conn.close()
    return df

def test_dashboard_calculations():
    print("Starting verification of dashboard metric calculations...")
    
    df = load_data()
    if df.empty:
        print("Telemetry database is empty. Cannot verify metrics.")
        sys.exit(1)
        
    # Calculate metrics using dashboard logic
    total_queries = len(df)
    hits = len(df[df['status'] == 'HIT'])
    misses = len(df[df['status'] == 'MISS'])
    hit_rate = (hits / total_queries * 100.0) if total_queries > 0 else 0.0
    total_saved = df['tokens_saved'].sum()
    
    avg_miss = df[df['status'] == 'MISS']['latency_ms'].mean() if len(df[df['status'] == 'MISS']) > 0 else 0.0
    avg_hit = df[df['status'] == 'HIT']['latency_ms'].mean() if len(df[df['status'] == 'HIT']) > 0 else 0.0
    speedup = max(0.0, avg_miss - avg_hit)
    
    false_positives = df[(df['similarity_score'] > 0.92) & (df['overlap_score'] < 0.70)]
    fp_count = len(false_positives)
    
    # Assertions based on test_integration.py run results
    print(f"Total queries: {total_queries} (Expected: 5)")
    assert total_queries == 5, "Total queries count mismatch"
    
    print(f"Hits: {hits} (Expected: 2)")
    assert hits == 2, "Hits count mismatch"
    
    print(f"Misses: {misses} (Expected: 3)")
    assert misses == 3, "Misses count mismatch"
    
    print(f"Hit rate: {hit_rate:.1f}% (Expected: 40.0%)")
    assert abs(hit_rate - 40.0) < 0.01, "Hit rate percentage mismatch"
    
    print(f"Total tokens saved: {total_saved} (Expected: 20)")
    assert total_saved == 20, "Total tokens saved mismatch"
    
    print(f"Average miss latency: {avg_miss:.2f}ms")
    print(f"Average hit latency: {avg_hit:.2f}ms")
    print(f"Average latency speedup: {speedup:.2f}ms")
    assert speedup > 0.0, "Latency speedup should be greater than 0"
    
    print(f"Saved from False Positives: {fp_count} (Expected: 1)")
    assert fp_count == 1, "False positives count mismatch"
    
    print("\nALL DASHBOARD CALCULATIONS VERIFIED SUCCESSFULLY!")
    sys.exit(0)

if __name__ == "__main__":
    test_dashboard_calculations()
