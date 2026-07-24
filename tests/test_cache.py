import os
import sys
import numpy as np

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.cache import SemanticCache

class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
        # Pre-determined semantic embedding vectors (normalized or close to normalized)
        self.vectors = {
            "What is the speed of a fast horse?": np.array([1.0, 0.0, 0.0]),
            "What is the speed of a quick horse?": np.array([0.95, 0.1, 0.0]),
            "What is the velocity of a swift equine?": np.array([0.93, 0.05, 0.0]),
            "How tall is a giraffe?": np.array([0.0, 1.0, 0.0])
        }

    def encode(self, query):
        if query in self.vectors:
            return self.vectors[query]
        # Return a default vector if not in dictionary
        return np.array([0.1, 0.1, 0.1])

def run_tests():
    print("Initializing test run...")
    
    # Instantiate the SemanticCache
    cache = SemanticCache()
    cache.clear()
    
    # Set thresholds to match test expectations
    import app.cache as cache_mod
    cache_mod.CACHE_HIT_THRESHOLD = 0.92
    cache_mod.OVERLAP_THRESHOLD = 0.70
    
    # Inject Mock SentenceTransformer to avoid loading model offline
    mock_model = MockSentenceTransformer('all-MiniLM-L6-v2')
    cache.model = mock_model
    
    # 1. Add query and response to the cache
    base_query = "What is the speed of a fast horse?"
    response = "Horses can run up to 55 mph."
    cache.add(base_query, response)
    
    success = True
    
    # 2. Test exact query match (should be a hit)
    print("\n--- Test 1: Exact Match ---")
    res1 = cache.lookup("What is the speed of a fast horse?")
    print(f"Result: {res1}")
    if res1 == response:
        print("Test 1 Passed (Exact Match Hit)")
    else:
        print("Test 1 Failed!")
        success = False
        
    # 3. Test near semantic match with high lexical overlap (should be a hit)
    # Overlap: 'What is the speed of a horse?' is in both. Overlap = 7/9 = 77.7% >= 70%.
    # Sim: 0.994 >= 0.92.
    print("\n--- Test 2: Semantic Match with High Lexical Overlap ---")
    res2 = cache.lookup("What is the speed of a quick horse?")
    print(f"Result: {res2}")
    if res2 == response:
        print("Test 2 Passed (Semantic Match Hit)")
    else:
        print("Test 2 Failed!")
        success = False

    # 4. Test near semantic match with low lexical overlap (should be a miss)
    # Overlap: 5/11 = 45.4% < 70%.
    # Sim: 0.998 >= 0.92.
    print("\n--- Test 3: Semantic Match with Low Lexical Overlap ---")
    res3 = cache.lookup("What is the velocity of a swift equine?")
    print(f"Result: {res3}")
    if res3 is None:
        print("Test 3 Passed (Lexical Check Blocked Cache Hit)")
    else:
        print("Test 3 Failed!")
        success = False

    # 5. Test completely different query (should be a miss)
    print("\n--- Test 4: Different Query ---")
    res4 = cache.lookup("How tall is a giraffe?")
    print(f"Result: {res4}")
    if res4 is None:
        print("Test 4 Passed (Different Query Miss)")
    else:
        print("Test 4 Failed!")
        success = False

    if success:
        print("\nALL SEMANTIC CACHE LOGIC TESTS PASSED!")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
