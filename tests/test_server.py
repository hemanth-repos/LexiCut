import os
import sys
import threading
import time
import urllib.request
import urllib.error
import json

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the main application
import app.main as main

def run_server(server):
    try:
        server.serve_forever()
    except Exception as e:
        print(f"Server exception: {e}")

def main_test():
    print("Starting verification test...")
    
    # Inject Mock SentenceTransformer and clear cache to avoid slow weights loading and timeouts
    from tests.test_cache import MockSentenceTransformer
    mock_model = MockSentenceTransformer('all-MiniLM-L6-v2')
    main.cache_instance.model = mock_model
    main.cache_instance.clear()
    
    print("Starting FastAPI/Uvicorn server in a thread...")
    import uvicorn
    config = uvicorn.Config(main.app, host="127.0.0.1", port=8085, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    port = 8085

    # Give server a moment to start
    time.sleep(2)

    success = True

    # Test GET /
    try:
        print(f"Testing GET request to http://127.0.0.1:{port}/...")
        req = urllib.request.Request(f"http://127.0.0.1:{port}/")
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            body = response.read().decode("utf-8")
            print(f"GET Status: {status}")
            print(f"GET Response Body: {body}")
            assert status == 200, "Root GET status code is not 200"
            res_json = json.loads(body)
            assert "libraries" in res_json, "Libraries list missing in response"
    except Exception as e:
        print(f"GET Request failed: {e}")
        success = False

    # Test POST /chat with valid payload
    try:
        print(f"Testing POST request to http://127.0.0.1:{port}/chat with valid query...")
        payload = {"query": "Hello Antigravity!"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            body = response.read().decode("utf-8")
            print(f"POST Chat Status: {status}")
            print(f"POST Chat Response Body: {body}")
            assert status == 200, "POST Chat status code is not 200"
            res_json = json.loads(body)
            assert "answer" in res_json, "Answer field missing in chat response"
            assert "Hello Antigravity!" in res_json["answer"], "Response content incorrect"
    except Exception as e:
        print(f"POST Chat Request failed: {e}")
        success = False

    # Test POST /chat with empty payload
    try:
        print(f"Testing POST request to http://127.0.0.1:{port}/chat with empty query...")
        payload = {"query": ""}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                print(f"Unexpected successful response for empty query: {response.status}")
                success = False
        except urllib.error.HTTPError as e:
            print(f"Received expected error status: {e.code}")
            assert e.code == 400, "Empty query did not return 400 Bad Request"
    except Exception as e:
        print(f"POST Empty Query test failed: {e}")
        success = False

    # Shutdown server
    print("Shutting down server thread...")
    server.should_exit = True

    if success:
        print("ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main_test()
