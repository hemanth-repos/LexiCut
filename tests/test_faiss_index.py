import os
import sys
import unittest
import numpy as np

# Ensure LexiCut root is in import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vector_index import FaissVectorIndex
from app.cache import SemanticCache
from tests.test_cache import MockSentenceTransformer

class TestFaissIndexAndCache(unittest.TestCase):
    def setUp(self):
        self.metadata_path = "test_metadata.pkl"
        self.index_path = "test_faiss.index"
        
        # Clean up files before test
        for path in [self.metadata_path, self.index_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def tearDown(self):
        # Clean up files after test
        for path in [self.metadata_path, self.index_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def test_add_vector(self):
        """Test adding vector to FaissVectorIndex."""
        idx = FaissVectorIndex(self.index_path)
        vec = np.array([0.5, 0.5, 0.7071], dtype=np.float32)
        idx.add_vector(query_id=42, embedding=vec)
        
        self.assertEqual(idx.dimension, 3)
        self.assertEqual(idx.index.ntotal, 1)

    def test_save_and_load_index(self):
        """Test saving and loading FAISS index file."""
        idx = FaissVectorIndex(self.index_path)
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        idx.add_vector(query_id=1, embedding=vec1)
        idx.add_vector(query_id=2, embedding=vec2)
        idx.save_index()
        
        self.assertTrue(os.path.exists(self.index_path))
        
        # Load in another index instance
        idx2 = FaissVectorIndex(self.index_path)
        self.assertEqual(idx2.dimension, 3)
        self.assertEqual(idx2.index.ntotal, 2)

    def test_top_k_retrieval(self):
        """Test retrieving top-k nearest neighbors."""
        idx = FaissVectorIndex(self.index_path)
        
        # Add 4 vectors with known similarities
        idx.add_vector(query_id=10, embedding=np.array([1.0, 0.0], dtype=np.float32))
        idx.add_vector(query_id=20, embedding=np.array([0.9, 0.435], dtype=np.float32))
        idx.add_vector(query_id=30, embedding=np.array([0.5, 0.866], dtype=np.float32))
        idx.add_vector(query_id=40, embedding=np.array([0.1, 0.995], dtype=np.float32))
        
        # Search using query close to [1.0, 0.0]
        query = np.array([1.0, 0.0], dtype=np.float32)
        distances, ids = idx.search(query, top_k=3)
        
        # Verify top-k length and descending order
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids[0], 10)
        self.assertEqual(ids[1], 20)
        self.assertEqual(ids[2], 30)
        
        # Distances are inner products of normalized vectors (cosine similarities)
        self.assertAlmostEqual(distances[0], 1.0, places=4)
        self.assertTrue(distances[0] >= distances[1] >= distances[2])

    def test_persistence_across_restart(self):
        """Test SemanticCache metadata and index persistence across restarts."""
        mock_model = MockSentenceTransformer('all-MiniLM-L6-v2')
        cache = SemanticCache(metadata_path=self.metadata_path, index_path=self.index_path, model=mock_model)
        
        # Add values
        cache.add("What is the speed of a fast horse?", "Horses run fast.")
        cache.add("How tall is a giraffe?", "Giraffes are tall.")
        
        # Re-instantiate cache
        cache2 = SemanticCache(metadata_path=self.metadata_path, index_path=self.index_path, model=mock_model)
        
        self.assertEqual(len(cache2.metadata), 2)
        self.assertEqual(cache2.vector_index.index.ntotal, 2)
        
        # Retrieve value from second instance
        res = cache2.lookup("What is the speed of a fast horse?")
        self.assertEqual(res, "Horses run fast.")

    def test_similarity_correctness(self):
        """Test that FAISS inner product matches manual NumPy cosine similarity."""
        idx = FaissVectorIndex(self.index_path)
        
        vec_a = np.array([0.123, 0.456, 0.789], dtype=np.float32)
        vec_b = np.array([0.987, 0.654, 0.321], dtype=np.float32)
        
        # Manual cosine similarity calculation
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        expected_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        
        idx.add_vector(query_id=100, embedding=vec_a)
        distances, ids = idx.search(vec_b, top_k=1)
        
        self.assertEqual(ids[0], 100)
        self.assertAlmostEqual(distances[0], expected_sim, places=5)

if __name__ == "__main__":
    unittest.main()
