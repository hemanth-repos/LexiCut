import os
import sys

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.cache import contains_antonym_conflict

# Automated tests for antonym detection
PASS_CASES = [
    ("Enable dark mode", "Disable dark mode"),
    ("Increase cache size", "Decrease cache size"),
    ("Install Docker", "Uninstall Docker"),
    ("Start server", "Stop server")
]

FAIL_CASES = [
    ("Write SQL query", "Generate SQL query"),
    ("Create database table", "Build database table"),
    ("Explain photosynthesis", "Describe photosynthesis")
]

def run_unit_tests():
    print("Starting unit tests for antonym conflict detection...")
    success = True
    
    # Verify PASS cases (should return True since contradiction exists)
    print("\n--- Verifying Contradiction Queries (Should Block / Return True) ---")
    for qa, qb in PASS_CASES:
        result = contains_antonym_conflict(qa, qb)
        print(f"[{qa}] vs [{qb}] => Result: {result}")
        if result is True:
            print("PASS - Successfully detected conflict.")
        else:
            print("FAIL - Failed to detect conflict.")
            success = False
            
    # Verify FAIL cases (should return False since no contradiction exists)
    print("\n--- Verifying Safe Queries (Should NOT Block / Return False) ---")
    for qa, qb in FAIL_CASES:
        result = contains_antonym_conflict(qa, qb)
        print(f"[{qa}] vs [{qb}] => Result: {result}")
        if result is False:
            print("PASS - Verified no false conflict triggers.")
        else:
            print("FAIL - Incorrectly triggered conflict block.")
            success = False
            
    if success:
        print("\nALL ANTONYM DETECTION UNIT TESTS PASSED!")
        sys.exit(0)
    else:
        print("\nSOME UNIT TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_unit_tests()
