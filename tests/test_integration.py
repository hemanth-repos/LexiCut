import os
import sys
import threading
import time
import urllib.request
import urllib.error
import json
import sqlite3
import numpy as np

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the main application
import app.main as main
from tests.test_cache import MockSentenceTransformer

def run_server(server):
    try:
        server.serve_forever()
    except Exception as e:
        print(f"Server exception: {e}")

def main_test():
    print("Starting integration test with SemanticCache...")
    
    # Set thresholds to match test expectations
    import app.cache as cache_mod
    cache_mod.CACHE_HIT_THRESHOLD = 0.92
    cache_mod.OVERLAP_THRESHOLD = 0.70
    
    # Inject Mock SentenceTransformer into the active cache_instance
    mock_model = MockSentenceTransformer('all-MiniLM-L6-v2')
    main.cache_instance.model = mock_model
    main.cache_instance.clear()

    # Run the server in a background thread
    print("Starting FastAPI/Uvicorn server...")
    import uvicorn
    config = uvicorn.Config(main.app, host="127.0.0.1", port=8085, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    port = 8085

    # Give server a moment to start
    time.sleep(2)

    success = True

    try:
        # Request 1: Initial query (Cache Miss)
        print("\n--- Request 1: Initial query (Should Miss) ---")
        payload = {"prompt": "What is the speed of a fast horse?"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
            print(f"Response: {body}")
            assert body.get("telemetry", {}).get("source") == "llm", "Expected source to be 'llm'"
            assert "Processed your query" in body.get("answer")

        # Request 2: Exact query again (Cache Hit)
        print("\n--- Request 2: Exact query again (Should Hit) ---")
        with urllib.request.urlopen(req, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
            print(f"Response: {body}")
            assert body.get("telemetry", {}).get("source") == "cache", "Expected source to be 'cache'"
            assert "tokens_saved" in body.get("telemetry", {})

        # Request 3: Semantically similar, high lexical overlap (Should Hit)
        print("\n--- Request 3: Similar query, high overlap (Should Hit) ---")
        payload = {"prompt": "What is the speed of a quick horse?"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
            print(f"Response: {body}")
            assert body.get("telemetry", {}).get("source") == "cache", "Expected source to be 'cache'"
            assert "What is the speed of a fast horse?" in body.get("answer")

        # Request 4: Semantically similar, low lexical overlap (Should Miss)
        print("\n--- Request 4: Similar query, low overlap (Should Miss) ---")
        payload = {"prompt": "What is the velocity of a swift equine?"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
            print(f"Response: {body}")
            assert body.get("telemetry", {}).get("source") == "llm", "Expected source to be 'llm' due to lexical block"

        # Request 5: Completely different query (Should Miss)
        print("\n--- Request 5: Different query (Should Miss) ---")
        payload = {"prompt": "How tall is a giraffe?"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
            print(f"Response: {body}")
            assert body.get("telemetry", {}).get("source") == "llm", "Expected source to be 'llm' for different query"

    except Exception as e:
        print(f"Assertion failed or connection error: {e}")
        success = False

    # Shutdown server
    print("\nShutting down server thread...")
    server.should_exit = True

    # Query SQLite database and print telemetry table contents
    print("\n--- Telemetry DB Logs ---")
    try:
        conn = sqlite3.connect("telemetry.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, prompt, status, similarity_score, overlap_score, latency_ms, tokens_used, tokens_saved FROM request_logs")
        rows = cursor.fetchall()
        for row in rows:
            print(f"ID: {row[0]} | Time: {row[1]} | Prompt: '{row[2]}' | Status: {row[3]} | Sim: {row[4]:.3f} | Overlap: {row[5]:.3f} | Latency: {row[6]:.2f}ms | Used: {row[7]} | Saved: {row[8]}")
        conn.close()
        
        if len(rows) < 5:
            print("Warning: Expected at least 5 log entries in telemetry database.")
            success = False
    except Exception as e:
        print(f"Error querying telemetry database: {e}")
        success = False

    if success:
        print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\nSOME INTEGRATION TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main_test()
