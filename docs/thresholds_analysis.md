# Threshold Evaluation and Optimization Report

This report analyzes the empirical results from running our test suite on adversarial and standard query pairs using `SentenceTransformer('all-MiniLM-L6-v2')` and Jaccard word overlap.

## Evaluation Results Table

| Pair | Similarity | Overlap | Expected | Classification (at Sim > 0.92, Overlap >= 0.30) |
| :--- | :--- | :--- | :--- | :--- |
| **Write a binary search in Python** vs **Can you code a binary search using Python?** | 0.9141 | 0.4000 | **HIT** | **False Miss** (Sim 0.9141 <= 0.92) |
| **Write a binary search in Python** vs **Implement binary search in Python** | 0.9473 | 0.5714 | **HIT** | **True Hit** (Sim > 0.92, Overlap >= 0.30) |
| **Write a binary search in Python** vs **Create a Python program for binary search** | 0.9251 | 0.4444 | **HIT** | **True Hit** (Sim > 0.92, Overlap >= 0.30) |
| **Write a binary search in Python** vs **What is machine learning?** | 0.1015 | 0.0000 | **MISS** | **True Miss** (Sim <= 0.92) |
| **Install Docker on Ubuntu** vs **Uninstall Docker from Ubuntu** | 0.7582 | 0.3333 | **MISS** | **True Miss** (Sim <= 0.92) |
| **Generate SQL query for customers** vs **Write SQL query for customers** | 0.9018 | 0.6667 | **HIT** | **False Miss** (Sim 0.9018 <= 0.92) |
| **Generate SQL query for customers** vs **Delete customer records from database** | 0.5111 | 0.0000 | **MISS** | **True Miss** (Sim <= 0.92) |
| **What is photosynthesis?** vs **Explain photosynthesis** | 0.9069 | 0.2500 | **HIT** | **False Miss** (Sim 0.9069 <= 0.92) |
| **What is photosynthesis?** vs **How does a car engine work?** | 0.2687 | 0.0000 | **MISS** | **True Miss** (Sim <= 0.92) |
| **Install Docker** vs **Uninstall Docker** (Adversarial) | 0.7881 | 0.3333 | **MISS** | **True Miss** (Sim <= 0.92) |
| **Enable dark mode** vs **Disable dark mode** (Adversarial) | 0.9437 | 0.5000 | **MISS** | **False Hit** (Sim 0.9437 > 0.92, Overlap 0.50 >= 0.30) |
| **Start server** vs **Stop server** (Adversarial) | 0.5805 | 0.3333 | **MISS** | **True Miss** (Sim <= 0.92) |
| **Increase cache size** vs **Decrease cache size** (Adversarial) | 0.9343 | 0.5000 | **MISS** | **False Hit** (Sim 0.9343 > 0.92, Overlap 0.50 >= 0.30) |

---

## Analysis of Classifications

Using the initial parameters (`CACHE_HIT_THRESHOLD = 0.92`, `OVERLAP_THRESHOLD = 0.30`):

### 1. True Hits (Correctly Cached)
- **Binary Search 2** (Sim: 0.9473, Overlap: 57.1%)
- **Binary Search 3** (Sim: 0.9251, Overlap: 44.4%)

### 2. False Hits (Incorrectly Cached - Antonym Leaks)
- **Enable dark mode vs Disable dark mode** (Sim: 0.9437, Overlap: 50.0%)
- **Increase cache size vs Decrease cache size** (Sim: 0.9343, Overlap: 50.0%)
- *Why it happened*: Semantic models map antonyms close together because they appear in identical contexts. A low overlap threshold of `0.30` allows them to leak through.

### 3. True Misses (Correctly Forwarded to LLM)
- All completely distinct queries (e.g., machine learning, car engine, etc.)
- Short antonym pairs like **Install vs Uninstall Docker** (Sim: 0.7881) and **Start vs Stop server** (Sim: 0.5805) because their semantic similarity fell below the `0.92` threshold.

### 4. False Misses (Valid hits that were blocked)
- **Binary Search 1** (Sim: 0.9141 - missed due to similarity threshold)
- **SQL Query** (Sim: 0.9018 - missed due to similarity threshold)
- **Photosynthesis** (Sim: 0.9069 - missed due to similarity threshold, and overlap of 25.0% is below 30%)

---

## Data-Driven Threshold Selection

To find the optimal threshold settings, we must look at the overlap and similarity distributions:

1. **Adversarial Misses (Antonym Leaks)** have:
   - Similarity: Up to `0.9437`
   - Overlap: Exactly `0.5000` (since they differ by only one keyword in a short query)
2. **Valid Hits** have:
   - Similarity: `0.9018` to `0.9473`
   - Overlap: `0.2500` to `0.6667`

### Strategy Options

#### Option A: Prioritize Accuracy (Prevent False Hits / Antonym Leaks)
To block all false hits, we must raise the lexical overlap threshold above `0.50`.
*   **Selected Thresholds**: `CACHE_HIT_THRESHOLD = 0.90`, `OVERLAP_THRESHOLD = 0.55`
*   **Impact**:
    - **False Hits**: 0% (All antonym leaks are successfully blocked).
    - **True Hits**: Only 2 out of 5 valid hits are cached (60% False Miss rate).
    - *Best for*: Safety-critical pipelines where returning an incorrect cached answer is unacceptable.

#### Option B: Prioritize Cache Hit Rate (Allow Varied Phrasings)
To allow more valid phrased queries through, we must keep the overlap threshold lower.
*   **Selected Thresholds**: `CACHE_HIT_THRESHOLD = 0.90`, `OVERLAP_THRESHOLD = 0.35`
*   **Impact**:
    - **True Hits**: 4 out of 5 valid hits are cached (80% True Hit rate).
    - **False Hits**: `Enable/Disable dark mode` and `Increase/Decrease cache size` will leak as false hits.
    - *Best for*: General conversational assistants where slightly context-imperfect hits are minor issues.

### Recommendation
For LexiCut, we recommend **Option A** (`CACHE_HIT_THRESHOLD = 0.90`, `OVERLAP_THRESHOLD = 0.55`) or implementing a **negation/antonym detection pre-filter** rather than relying purely on simple Jaccard overlap, because antonym leaks (like starting vs. stopping a server) are functionally dangerous for system automation pipelines.
