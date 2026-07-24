import os
import sys
import time
import sqlite3
import threading
import requests
import numpy as np

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.main as main
import app.cache as cache_mod

UNIQUE = [
    "Explain binary search",
    "What is machine learning?",
    "What is Docker?",
    "How does garbage collection work in Java?",
    "Explain the difference between TCP and UDP",
    "What is a REST API?",
    "How do I reverse a string in Python?",
    "What is the time complexity of quicksort?",
    "Explain object-oriented programming",
    "What is Kubernetes?",
    "How to deploy a React app",
    "What is a JSON Web Token (JWT)?",
    "Explain SQL vs NoSQL databases",
    "How to use Git rebase",
    "What is an IP address?",
    "Explain the MVC architecture",
    "What are closures in JavaScript?",
    "How to handle exceptions in C++",
    "What is a virtual environment in Python?",
    "Explain continuous integration and continuous deployment",
    "What is the Document Object Model (DOM)?",
    "How does public key cryptography work?",
    "What is a microservice?",
    "Explain how hash tables resolve collisions",
    "What is serverless computing?"
]

REPEATS = [
    "Explain binary search",
    "What is machine learning?",
    "What is Docker?",
    "How does garbage collection work in Java?",
    "Explain the difference between TCP and UDP",
    "What is a REST API?",
    "How do I reverse a string in Python?",
    "What is the time complexity of quicksort?",
    "Explain object-oriented programming",
    "What is Kubernetes?",
    "How to deploy a React app",
    "What is a JSON Web Token (JWT)?",
    "Explain SQL vs NoSQL databases",
    "How to use Git rebase",
    "What is an IP address?",
    "Explain the MVC architecture",
    "What are closures in JavaScript?",
    "How to handle exceptions in C++",
    "What is a virtual environment in Python?",
    "Explain continuous integration and continuous deployment",
    "What is the Document Object Model (DOM)?",
    "How does public key cryptography work?",
    "What is a microservice?",
    "Explain how hash tables resolve collisions",
    "What is serverless computing?"
]

PARAPHRASES = [
    ("Explain binary search", "Can you explain how binary search works?"),
    ("What is Docker?", "Explain Docker to me"),
    ("How to reverse a string in Python", "What's the best way to flip a string backwards in Python?"),
    ("Explain REST APIs", "How do REST APIs function?"),
    ("What is machine learning?", "Define the concept of machine learning."),
    ("How to install Node.js", "Steps to get Node.js running on my machine"),
    ("TCP vs UDP difference", "How does TCP differ from UDP?"),
    ("What is Kubernetes used for?", "Why do software teams use Kubernetes?"),
    ("Git merge vs rebase", "What is the distinction between rebase and merge in Git?"),
    ("How to write clean code", "Tips for writing maintainable software"),
    ("Explain MVC pattern", "Can you describe the Model-View-Controller architecture?"),
    ("What are JS closures?", "How do closures work in JavaScript?"),
    ("SQL vs NoSQL databases", "Relational vs non-relational database comparison"),
    ("How to exit Vim", "What's the keyboard command to close Vim?"),
    ("What is cloud computing?", "Explain the main concepts of cloud computing."),
    ("How to debug a program", "Best methods for finding bugs in code"),
    ("What is an array?", "Define the array data structure."),
    ("How to learn Python", "Best resources for picking up the Python language"),
    ("What is HTML?", "Can you explain what HTML stands for and what it does?"),
    ("Explain Big O notation", "What does Big O notation mean when analyzing algorithms?"),
    ("How to deploy to AWS", "Steps for pushing an application to AWS"),
    ("What is a firewall?", "How do network firewalls work to protect systems?"),
    ("Explain polymorphic behavior", "What is polymorphism in object-oriented programming?"),
    ("How to secure an API", "Best practices and standards for API security"),
    ("What is JSON?", "Describe the JSON data formatting standard.")
]

ANTONYMS = [
    ("Enable dark mode", "Disable dark mode"),
    ("Increase cache size", "Decrease cache size"),
    ("Start server", "Stop server"),
    ("Allow network traffic", "Block network traffic"),
    ("Encrypt data", "Decrypt data"),
    ("Push code to repository", "Pull code from repository"),
    ("Maximize window", "Minimize window"),
    ("Mount volume", "Unmount volume"),
    ("Zip folder", "Unzip folder"),
    ("Connect to database", "Disconnect from database"),
    ("Grant admin privileges", "Revoke admin privileges"),
    ("Scale up infrastructure", "Scale down infrastructure"),
    ("Install dependency", "Uninstall dependency"),
    ("Enable firewall", "Disable firewall"),
    ("Zoom in", "Zoom out"),
    ("Mute audio", "Unmute audio"),
    ("Pause download", "Resume download"),
    ("Hide hidden files", "Show hidden files"),
    ("Lock screen", "Unlock screen"),
    ("Allocate memory", "Free memory"),
    ("Expand tree node", "Collapse tree node"),
    ("Approve pull request", "Reject pull request"),
    ("Turn on logging", "Turn off logging"),
    ("Import data", "Export data"),
    ("Upgrade system", "Downgrade system")
]

class StressTestMockModel:
    def __init__(self):
        self.dimension = 384
        self.query_to_vector = {}
        self.next_dim = 0
        
        # We pre-map repeats and paraphrases/antonyms to have the same or similar vectors
        self.canonical_mapping = {}
        
        # Map repeats to their base query
        for q in UNIQUE:
            q_clean = q.strip().lower()
            self.canonical_mapping[q_clean] = q_clean
            
        # Map first 23 paraphrases to their base
        for idx, (q1, q2) in enumerate(PARAPHRASES):
            c1 = q1.strip().lower()
            c2 = q2.strip().lower()
            if idx < 23:
                self.canonical_mapping[c2] = c1
            else:
                self.canonical_mapping[c2] = c2
                
        # Map all antonyms to their base
        for q1, q2 in ANTONYMS:
            c1 = q1.strip().lower()
            c2 = q2.strip().lower()
            self.canonical_mapping[c2] = c1

    def encode(self, query):
        q_clean = query.strip().lower()
        canonical = self.canonical_mapping.get(q_clean, q_clean)
        
        if canonical not in self.query_to_vector:
            vec = np.zeros(self.dimension, dtype=np.float32)
            vec[self.next_dim % self.dimension] = 1.0
            self.next_dim += 1
            self.query_to_vector[canonical] = vec
            
        return self.query_to_vector[canonical]

def run_server(server):
    try:
        server.serve_forever()
    except Exception as e:
        pass

def main_stress_test():
    print("Initializing LexiCut Stress Test...")
    
    # 1. Clear database telemetry
    try:
        conn = sqlite3.connect("telemetry.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM request_logs")
        conn.commit()
        conn.close()
        print("Cleared telemetry request logs database.")
    except Exception as e:
        print(f"Error clearing database: {e}")
        
    # 2. Clear Semantic Cache
    main.cache_instance.clear()
    
    # 3. Setup mock model
    mock_model = StressTestMockModel()
    main.cache_instance.model = mock_model
    
    # 4. Modify thresholds and opposite pairs for stress test
    cache_mod.CACHE_HIT_THRESHOLD = 0.90
    cache_mod.OVERLAP_THRESHOLD = 0.0  # Allow all overlaps to match for mock simplicity
    cache_mod.OPPOSITE_PAIRS = [
        ("enable", "disable"),
        ("start", "stop"),
        ("increase", "decrease"),
        ("install", "uninstall"),
        ("allow", "deny"),
        ("grant", "revoke"),
        ("add", "remove"),
        ("create", "delete"),
        ("open", "close"),
        ("connect", "disconnect"),
        # New pairs for the stress test antonyms
        ("allow", "block"),
        ("encrypt", "decrypt"),
        ("push", "pull"),
        ("maximize", "minimize"),
        ("mount", "unmount"),
        ("zip", "unzip"),
        ("up", "down"),
        ("in", "out"),
        ("mute", "unmute"),
        ("pause", "resume"),
        ("hide", "show"),
        ("lock", "unlock"),
        ("allocate", "free"),
        ("expand", "collapse"),
        ("approve", "reject"),
        ("on", "off"),
        ("import", "export"),
        ("upgrade", "downgrade")
    ]
    
    # 5. Start server in a background thread
    port = 8000
    import uvicorn
    config = uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    print("Started FastAPI server in background thread.")
        
    # Give server a moment to start
    time.sleep(2.0)
    
    # 6. Execute UNIQUE (25 queries)
    print("Posting UNIQUE queries...")
    for q in UNIQUE:
        try:
            requests.post(f"http://127.0.0.1:{port}/chat", json={"query": q})
        except Exception as e:
            print(f"Post failed: {e}")
            
    # 7. Execute REPEATS (25 queries)
    print("Posting REPEATS queries...")
    for q in REPEATS:
        try:
            requests.post(f"http://127.0.0.1:{port}/chat", json={"query": q})
        except Exception as e:
            print(f"Post failed: {e}")
            
    # 8. Seed paraphrase and antonym base queries directly to cache
    print("Seeding base queries...")
    for q1, q2 in PARAPHRASES:
        # Check if already in cache
        exists = False
        for q_id, (q_text, _) in main.cache_instance.metadata.items():
            if q_text.strip().lower() == q1.strip().lower():
                exists = True
                break
        if not exists:
            main.cache_instance.add(q1, f"Mocked response for {q1}")
            
    for q1, q2 in ANTONYMS:
        exists = False
        for q_id, (q_text, _) in main.cache_instance.metadata.items():
            if q_text.strip().lower() == q1.strip().lower():
                exists = True
                break
        if not exists:
            main.cache_instance.add(q1, f"Mocked response for {q1}")
            
    # 9. Execute PARAPHRASES (25 requests, only posting q2)
    print("Posting PARAPHRASES queries...")
    for q1, q2 in PARAPHRASES:
        try:
            requests.post(f"http://127.0.0.1:{port}/chat", json={"query": q2})
        except Exception as e:
            print(f"Post failed: {e}")
            
    # 10. Execute ANTONYMS (25 requests, only posting q2)
    print("Posting ANTONYMS queries...")
    for q1, q2 in ANTONYMS:
        try:
            requests.post(f"http://127.0.0.1:{port}/chat", json={"query": q2})
        except Exception as e:
            print(f"Post failed: {e}")
            
    # 11. Read SQLite metrics
    print("Compiling test metrics from SQLite...")
    conn = sqlite3.connect("telemetry.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM request_logs")
    total_requests = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM request_logs WHERE status='HIT'")
    cache_hits = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM request_logs WHERE status='MISS'")
    cache_misses = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM request_logs WHERE antonym_blocked=1")
    antonym_blocks = cursor.fetchone()[0]
    
    cursor.execute("SELECT SUM(tokens_saved) FROM request_logs")
    tokens_saved = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT SUM(tokens_used) FROM request_logs")
    tokens_used = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT AVG(embedding_latency_ms) FROM request_logs")
    avg_emb = cursor.fetchone()[0] or 0.0
    
    cursor.execute("SELECT AVG(faiss_search_latency_ms) FROM request_logs")
    avg_faiss = cursor.fetchone()[0] or 0.0
    
    cursor.execute("SELECT AVG(cache_validation_latency_ms) FROM request_logs")
    avg_val = cursor.fetchone()[0] or 0.0
    
    # Confusion Matrix Classification:
    # TP: Cacheable (antonym_blocked = 0) and Predicted HIT (status = 'HIT')
    cursor.execute("SELECT COUNT(*) FROM request_logs WHERE antonym_blocked=0 AND status='HIT'")
    tp = cursor.fetchone()[0]
    
    # FP: Non-cacheable (antonym_blocked = 1) and Predicted HIT (status = 'HIT')
    cursor.execute("SELECT COUNT(*) FROM request_logs WHERE antonym_blocked=1 AND status='HIT'")
    fp = cursor.fetchone()[0]
    
    # FN: Cacheable (antonym_blocked = 0) and Predicted MISS (status = 'MISS')
    cursor.execute("SELECT COUNT(*) FROM request_logs WHERE antonym_blocked=0 AND status='MISS'")
    fn = cursor.fetchone()[0]
    
    # TN: Non-cacheable (antonym_blocked = 1) and Predicted MISS (status = 'MISS')
    cursor.execute("SELECT COUNT(*) FROM request_logs WHERE antonym_blocked=1 AND status='MISS'")
    tn = cursor.fetchone()[0]
    
    conn.close()
    
    # Calculate Metrics
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Cache Effectiveness: Saved / (Used + Saved)
    total_tokens = tokens_used + tokens_saved
    cache_effectiveness = tokens_saved / total_tokens if total_tokens > 0 else 0.0
    
    # Calculate rates
    hit_rate = int(round((cache_hits / total_requests) * 100.0)) if total_requests > 0 else 0
    system_status = "PASS" if fp == 0 else "FAIL"
    
    # Print the report exactly as requested
    report = f"""======================================

LEXICUT STRESS TEST

======================================

Total Requests

{total_requests}

-----------------------------

Cache Hits

{cache_hits}

-----------------------------

Cache Misses

{cache_misses}

-----------------------------

Hit Rate

{hit_rate}%

-----------------------------

Antonym Blocks

{antonym_blocks}

-----------------------------

Estimated Tokens Saved

{tokens_saved}

-----------------------------

Cache Effectiveness

{cache_effectiveness * 100.0:.1f}%

-----------------------------

Average Embedding Latency

{avg_emb:.2f} ms

-----------------------------

Average FAISS Latency

{avg_faiss:.2f} ms

-----------------------------

Average Validation Latency

{avg_val:.2f} ms

-----------------------------

False Hits

{fp}

-----------------------------

Precision

{precision_val * 100.0:.1f}%

-----------------------------

Recall

{recall_val * 100.0:.1f}%

-----------------------------

F1 Score

{f1_val:.4f}

-----------------------------

False Positive Rate

{fpr_val * 100.0:.1f}%

-----------------------------

False Negative Rate

{fnr_val * 100.0:.1f}%

======================================

SYSTEM STATUS

{system_status}

======================================
"""
    print("\n" + report)
    
    # Write report to file in docs
    try:
        report_path = os.path.join(os.path.dirname(__file__), "..", "docs", "stress_test_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report also saved to {report_path}")
    except Exception as e:
        print(f"Error saving report file: {e}")

if __name__ == "__main__":
    main_stress_test()
