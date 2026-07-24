import os
import sys
import re
import numpy as np

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer  # type: ignore

TEST_QUERIES = [
    (
        "Write a binary search in Python",
        "Can you code a binary search using Python?",
        "HIT"
    ),
    (
        "Write a binary search in Python",
        "Implement binary search in Python",
        "HIT"
    ),
    (
        "Write a binary search in Python",
        "Create a Python program for binary search",
        "HIT"
    ),
    (
        "Write a binary search in Python",
        "What is machine learning?",
        "MISS"
    ),
    (
        "Install Docker on Ubuntu",
        "Uninstall Docker from Ubuntu",
        "MISS"
    ),
    (
        "Generate SQL query for customers",
        "Write SQL query for customers",
        "HIT"
    ),
    (
        "Generate SQL query for customers",
        "Delete customer records from database",
        "MISS"
    ),
    (
        "What is photosynthesis?",
        "Explain photosynthesis",
        "HIT"
    ),
    (
        "What is photosynthesis?",
        "How does a car engine work?",
        "MISS"
    ),
    # Hard cases
    (
        "Install Docker",
        "Uninstall Docker",
        "MISS"
    ),
    (
        "Enable dark mode",
        "Disable dark mode",
        "MISS"
    ),
    (
        "Start server",
        "Stop server",
        "MISS"
    ),
    (
        "Increase cache size",
        "Decrease cache size",
        "MISS"
    )
]

def compute_word_overlap(query1: str, query2: str) -> float:
    words1 = set(re.findall(r'\w+', query1.lower()))
    words2 = set(re.findall(r'\w+', query2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)

def main():
    print("Loading SentenceTransformer('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    results = []
    
    for q_a, q_b, expected in TEST_QUERIES:
        # Compute embeddings
        emb_a = model.encode(q_a)
        emb_b = model.encode(q_b)
        
        # Compute cosine similarity
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a == 0 or norm_b == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
            
        # Compute lexical overlap
        overlap = compute_word_overlap(q_a, q_b)
        
        # Print requested format
        print("=================================================")
        print(f"{q_a}")
        print(f"{q_b}")
        print()
        print(f"Similarity: {similarity:.4f}")
        print(f"Overlap: {overlap:.4f}")
        print()
        print(f"Expected: {expected}")
        print("=================================================")
        
        pair_name = f"{q_a} vs {q_b}"
        results.append({
            "pair": pair_name,
            "similarity": similarity,
            "overlap": overlap,
            "expected": expected
        })
        
    # Generate Markdown Table contents
    markdown_lines = []
    markdown_lines.append("# Threshold Evaluation Results\n")
    markdown_lines.append("| Pair | Similarity | Overlap | Expected |")
    markdown_lines.append("| --- | --- | --- | --- |")
    for r in results:
        markdown_lines.append(f"| {r['pair']} | {r['similarity']:.4f} | {r['overlap']:.4f} | {r['expected']} |")
        
    report_path = os.path.join(os.path.dirname(__file__), "..", "docs", "thresholds_evaluation.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))
        
    print(f"\nEvaluation complete! Results written to {report_path}.")

if __name__ == "__main__":
    main()
