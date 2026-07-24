import re
import os
import sys
import time
import pickle
import numpy as np

# Ensure root directory is in sys.path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from app.vector_index import FaissVectorIndex

CACHE_HIT_THRESHOLD = 0.90
OVERLAP_THRESHOLD = 0.30

OPPOSITE_PAIRS = [
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

def contains_antonym_conflict(query_a: str, query_b: str) -> bool:
    """
    Detects if query_a and query_b have opposite terms appearing across them.
    Returns True if a contradiction/antonym conflict exists.
    """
    words_a = set(re.findall(r'\w+', query_a.lower()))
    words_b = set(re.findall(r'\w+', query_b.lower()))
    
    for term1, term2 in OPPOSITE_PAIRS:
        if (term1 in words_a and term2 in words_b) or (term2 in words_a and term1 in words_b):
            return True
            
    return False

class SemanticCache:
    def __init__(self, metadata_path: str = "query_metadata.pkl", index_path: str = "faiss.index", model=None):
        self.metadata_path = metadata_path
        self.index_path = index_path
        self.model = model
        
        # Load metadata dictionary (query_id -> (query_text, response_text))
        self.metadata = {}
        self._load_metadata()
        
        # Initialize FAISS Index
        self.vector_index = FaissVectorIndex(self.index_path)
        
        # Gracefully rebuild the FAISS index if metadata exists but index is empty/missing/mismatched
        self._rebuild_index_if_needed()

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded cache metadata from {self.metadata_path} with {len(self.metadata)} entries.")
            except Exception as e:
                print(f"Error loading metadata from {self.metadata_path}: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            print(f"Saved cache metadata to {self.metadata_path}")
        except Exception as e:
            print(f"Error saving metadata to {self.metadata_path}: {e}")

    def _rebuild_index_if_needed(self):
        has_metadata = len(self.metadata) > 0
        index_empty_or_missing = (self.vector_index.index is None or self.vector_index.index.ntotal == 0)
        
        mismatch = False
        if self.vector_index.index is not None:
            if self.vector_index.index.ntotal != len(self.metadata):
                mismatch = True
                
        if has_metadata and (index_empty_or_missing or mismatch):
            print("Gracefully rebuilding FAISS index from metadata pickle...")
            if os.path.exists(self.index_path):
                try:
                    os.remove(self.index_path)
                except Exception:
                    pass
            self.vector_index = FaissVectorIndex(self.index_path)
            
            # Re-encode and rebuild index
            model = self._get_model()
            for query_id, (query, response) in self.metadata.items():
                try:
                    embedding = np.array(model.encode(query), dtype=np.float32)
                    self.vector_index.add_vector(query_id, embedding)
                except Exception as e:
                    print(f"Error encoding query {query_id} during rebuild: {e}")
            
            self.vector_index.save_index()
            print(f"Rebuild completed. Reindexed {self.vector_index.index.ntotal if self.vector_index.index else 0} entries.")

    def clear(self):
        """Clear all entries in the metadata store and the FAISS index."""
        self.metadata = {}
        if os.path.exists(self.metadata_path):
            try:
                os.remove(self.metadata_path)
            except Exception as e:
                print(f"Error removing metadata file on clear: {e}")
                
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
            except Exception as e:
                print(f"Error removing FAISS index file on clear: {e}")
        
        # Reinitialize the vector index
        self.vector_index = FaissVectorIndex(self.index_path)

    def _get_model(self):
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model

    def add(self, query: str, response: str):
        """Add a query and its response to the cache."""
        try:
            model = self._get_model()
            embedding = model.encode(query)
            embedding = np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error encoding query '{query}': {e}")
            embedding = None

        # Check if the exact query text is already in metadata
        query_id = None
        for q_id, (q_text, r_text) in self.metadata.items():
            if q_text == query:
                query_id = q_id
                break
                
        if query_id is not None:
            # Update matching entry
            self.metadata[query_id] = (query, response)
            self._save_metadata()
            
            if embedding is not None:
                self.vector_index.add_vector(query_id, embedding)
                self.vector_index.save_index()
        else:
            # Create a new query entry
            query_id = max(self.metadata.keys()) + 1 if self.metadata else 1
            self.metadata[query_id] = (query, response)
            self._save_metadata()
            
            if embedding is not None:
                self.vector_index.add_vector(query_id, embedding)
                self.vector_index.save_index()

    def check(self, prompt: str):
        """
        Check if the prompt exists in the cache, verifying both semantic and lexical thresholds,
        as well as checking for semantic antonym conflicts.
        
        Returns a 10-tuple:
        (
            cached_answer,
            similarity_score,
            overlap_score,
            antonym_blocked: bool,
            candidates_examined: int,
            faiss_similarity: float,
            retrieval_method: str,
            embedding_latency_ms: float,
            faiss_search_latency_ms: float,
            cache_validation_latency_ms: float
        )
        cached_answer will be None if it's a miss or if blocked.
        """
        if not self.metadata:
            return None, 0.0, 0.0, False, 0, 0.0, "faiss", 0.0, 0.0, 0.0

        # Encode new query
        try:
            model = self._get_model()
            embedding_start = time.perf_counter()
            query_embedding = np.array(model.encode(prompt), dtype=np.float32)
            embedding_latency = (time.perf_counter() - embedding_start) * 1000.0
        except Exception as e:
            print(f"Could not encode query '{prompt}': {e}")
            # Offline/fallback exact match cache logic in query_metadata.pkl
            start_scan = time.perf_counter()
            candidates_examined = 0
            for q_id, (q_text, r_text) in self.metadata.items():
                candidates_examined += 1
                if q_text.strip().lower() == prompt.strip().lower():
                    scan_latency = (time.perf_counter() - start_scan) * 1000.0
                    # For exact match, similarity and overlap are 1.0, and no antonym conflict possible
                    return r_text, 1.0, 1.0, False, candidates_examined, 1.0, "linear_scan", 0.0, 0.0, scan_latency
            scan_latency = (time.perf_counter() - start_scan) * 1000.0
            return None, 0.0, 0.0, False, candidates_examined, 0.0, "linear_scan", 0.0, 0.0, scan_latency

        # Search the FAISS vector index
        faiss_start = time.perf_counter()
        distances, ids = self.vector_index.search(query_embedding, top_k=5)
        faiss_latency = (time.perf_counter() - faiss_start) * 1000.0
        
        if not ids:
            return None, 0.0, 0.0, False, 0, 0.0, "faiss", embedding_latency, faiss_latency, 0.0

        # Keep track of metrics for the best candidate (first one returned by FAISS) for miss telemetry reporting
        first_similarity = distances[0]
        first_id = ids[0]
        first_overlap = 0.0
        
        start_validation = time.perf_counter()
        if first_id in self.metadata:
            first_overlap = self._compute_word_overlap(prompt, self.metadata[first_id][0])

        antonym_blocked_any = False
        candidates_examined = 0
        hit_found = False
        hit_response = None
        hit_similarity = 0.0
        hit_overlap = 0.0
        hit_query_id = None
        hit_candidate_query = None

        # Evaluate each candidate in descending order of similarity
        for similarity, query_id in zip(distances, ids):
            print(
                f"Candidate={query_id}, "
                f"Similarity={similarity:.4f}"
            )

            candidates_examined += 1
            # 1. Similarity Validation
            if similarity <= CACHE_HIT_THRESHOLD:
                # Since distances are sorted descending, subsequent candidates will also fail similarity validation
                break
                
            if query_id not in self.metadata:
                continue

            candidate_query, candidate_response = self.metadata[query_id]
            
            # 2. Lexical Overlap Validation
            overlap = self._compute_word_overlap(prompt, candidate_query)
            print(
                f"Candidate={query_id}, "
                f"Similarity={similarity:.4f}, "
                f"Overlap={overlap:.4f}"
            )
            
            if overlap >= OVERLAP_THRESHOLD:
                # 3. Antonym Safety Validation
                if contains_antonym_conflict(prompt, candidate_query):
                    print(f"Cache Blocked (Candidate {query_id}): Antonym conflict between '{prompt}' and '{candidate_query}'")
                    antonym_blocked_any = True
                    # Do not return immediately; check other candidates in case one of them is a valid hit
                else:
                    # Return first valid cache hit!
                    hit_found = True
                    hit_response = candidate_response
                    hit_similarity = similarity
                    hit_overlap = overlap
                    hit_query_id = query_id
                    hit_candidate_query = candidate_query
                    break

        validation_latency = (time.perf_counter() - start_validation) * 1000.0

        if hit_found:
            print(f"Lookup query: '{prompt}' -> HIT (Candidate {hit_query_id}): '{hit_candidate_query}' with similarity: {hit_similarity:.4f}, overlap: {hit_overlap:.4f}")
            return hit_response, hit_similarity, hit_overlap, False, candidates_examined, first_similarity, "faiss", embedding_latency, faiss_latency, validation_latency
        else:
            return None, first_similarity, first_overlap, antonym_blocked_any, candidates_examined, first_similarity, "faiss", embedding_latency, faiss_latency, validation_latency

    def lookup(self, query: str):
        """
        Look up a query in the cache.
        Returns the cached response if all threshold and safety checks pass, otherwise None.
        """
        response, _, _, _, _, _, _, _, _, _ = self.check(query)
        return response

    def _compute_word_overlap(self, query1: str, query2: str) -> float:
        # Extract lowercase alphanumeric words
        words1 = set(re.findall(r'\w+', query1.lower()))
        words2 = set(re.findall(r'\w+', query2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
