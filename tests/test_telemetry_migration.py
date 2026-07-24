import os
import sys
import sqlite3
import unittest
import numpy as np

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.main as main
from app.cache import SemanticCache
from tests.test_cache import MockSentenceTransformer

class TestTelemetryMigration(unittest.TestCase):
    def setUp(self):
        # We will use a separate test telemetry database for isolation
        self.db_path = "test_telemetry.db"
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
                
        # Save a reference to the original connect method to prevent recursion
        import sqlite3 as orig_sqlite3
        self._real_connect = orig_sqlite3.connect
        
        # Monkeypatch connection paths in main module
        main.sqlite3.connect = self._mock_connect
        
    def tearDown(self):
        # Restore mock patches
        import sqlite3 as orig_sqlite3
        main.sqlite3.connect = orig_sqlite3.connect
        
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def _mock_connect(self, database, *args, **kwargs):
        # Force all connections inside main module to point to our test database
        if database == "telemetry.db":
            return self._real_connect(self.db_path, *args, **kwargs)
        return self._real_connect(database, *args, **kwargs)

    def test_db_migration_and_creation_succeeds(self):
        """Verify that telemetry table creation and migration scripts succeed on empty and pre-existing schemas."""
        # 1. Test clean creation of table and all columns
        main.init_db()
        
        conn = self._real_connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(request_logs)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()
        
        # Assert crucial new telemetry columns are present
        self.assertIn("embedding_latency_ms", columns)
        self.assertIn("faiss_search_latency_ms", columns)
        self.assertIn("cache_validation_latency_ms", columns)
        self.assertIn("faiss_retrieval_latency_ms", columns)
        
        # 2. Test migration on top of a legacy database schema (without the new columns)
        # Clean up database
        os.remove(self.db_path)
        
        # Initialize legacy database table
        conn = self._real_connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                status TEXT,
                similarity_score REAL,
                overlap_score REAL,
                latency_ms REAL,
                tokens_used INTEGER,
                tokens_saved INTEGER
            )
        """)
        conn.commit()
        conn.close()
        
        # Run init_db() migration script
        main.init_db()
        
        # Check that columns were added dynamically
        conn = self._real_connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(request_logs)")
        columns_migrated = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()
        
        self.assertIn("embedding_latency_ms", columns_migrated)
        self.assertIn("faiss_search_latency_ms", columns_migrated)
        self.assertIn("cache_validation_latency_ms", columns_migrated)
        self.assertIn("faiss_retrieval_latency_ms", columns_migrated)

    def test_insert_succeeds(self):
        """Verify that inserting request telemetry via log_request runs successfully and stores values."""
        main.init_db()
        
        # Insert test request telemetry
        main.log_request(
            prompt="Test query prompt",
            status="HIT",
            similarity_score=0.95,
            overlap_score=0.85,
            latency_ms=12.5,
            tokens_used=0,
            tokens_saved=50,
            antonym_blocked=0,
            faiss_candidates_examined=3,
            faiss_similarity=0.95,
            retrieval_method="faiss",
            faiss_retrieval_latency_ms=8.2,
            cache_validation_latency_ms=4.3,
            embedding_latency_ms=6.1,
            faiss_search_latency_ms=2.1
        )
        
        conn = self._real_connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT prompt, status, embedding_latency_ms, faiss_search_latency_ms, cache_validation_latency_ms 
            FROM request_logs 
            ORDER BY id DESC LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "Test query prompt")
        self.assertEqual(row[1], "HIT")
        self.assertEqual(row[2], 6.1)
        self.assertEqual(row[3], 2.1)
        self.assertEqual(row[4], 4.3)

    def test_dashboard_queries_succeed(self):
        """Verify that dashboard aggregation queries (averages) execute perfectly on the migrated schema."""
        main.init_db()
        
        # Insert a couple of logs
        main.log_request("Prompt 1", "HIT", 0.95, 0.90, 10.0, 0, 40, 0, 1, 0.95, "faiss", 8.0, 2.0, 6.0, 2.0)
        main.log_request("Prompt 2", "MISS", 0.10, 0.05, 120.0, 50, 0, 0, 1, 0.10, "faiss", 9.0, 1.0, 7.0, 2.0)
        
        conn = self._real_connect(self.db_path)
        cursor = conn.cursor()
        
        # Emulate queries that dashboard runs:
        # 1. Total records, hit count, etc.
        cursor.execute("SELECT * FROM request_logs ORDER BY id ASC")
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 2)
        
        # 2. Search Performance Averages
        cursor.execute("""
            SELECT 
                AVG(embedding_latency_ms), 
                AVG(faiss_search_latency_ms), 
                AVG(cache_validation_latency_ms) 
            FROM request_logs
        """)
        avg_emb, avg_search, avg_val = cursor.fetchone()
        
        # Assert averages are correctly calculated
        self.assertEqual(avg_emb, 6.5)
        self.assertEqual(avg_search, 2.0)
        self.assertEqual(avg_val, 1.5)
        
        conn.close()

    def test_cache_check_tuple_length(self):
        """Verify that SemanticCache.check() returns the correct 10-tuple format and lengths."""
        mock_model = MockSentenceTransformer('all-MiniLM-L6-v2')
        cache = SemanticCache(metadata_path="test_meta.pkl", index_path="test_faiss.index", model=mock_model)
        cache.clear()
        
        # Add a mock entry
        cache.add("What is the speed of a fast horse?", "Horses run fast.")
        
        # Call check()
        result = cache.check("What is the speed of a quick horse?")
        
        # Assert it returns a 10-tuple
        self.assertEqual(len(result), 10)
        
        # Unpack and verify elements
        (
            cached_answer,
            similarity_score,
            overlap_score,
            antonym_blocked,
            candidates_examined,
            faiss_similarity,
            retrieval_method,
            embedding_latency_ms,
            faiss_search_latency_ms,
            cache_validation_latency_ms
        ) = result
        
        # Clean up files
        cache.clear()
        for path in ["test_meta.pkl", "test_faiss.index"]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    unittest.main()
