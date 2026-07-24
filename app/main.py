import os
import sys
import sqlite3
import time
import numpy as np
import sklearn
import sentence_transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Ensure root directory is in sys.path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import google.antigravity SDK
try:
    from google.antigravity import Agent, LocalAgentConfig  # type: ignore
    HAS_ANTIGRAVITY_SDK = True
except ImportError:
    HAS_ANTIGRAVITY_SDK = False

# Mock classes for offline fallback if google-antigravity SDK is not installed
if not HAS_ANTIGRAVITY_SDK:
    class LocalAgentConfig:
        def __init__(self, system_instructions=None):
            self.system_instructions = system_instructions

    class MockSDKResponse:
        def __init__(self, text_content):
            self._text = text_content
            # Simulating usage metadata structure from Gemini API
            self.usage_metadata = type('obj', (object,), {
                'prompt_token_count': 15,
                'candidates_token_count': 25,
                'total_token_count': 40
            })()

        async def text(self):
            return self._text

    class Agent:
        def __init__(self, config=None):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def chat(self, prompt: str):
            # Fallback mock response for the offline sandbox
            return MockSDKResponse(f"Processed your query: '{prompt}'")

# Import SemanticCache from app package
from app.cache import SemanticCache

cache_instance = SemanticCache()

def init_db():
    """Auto-initialize telemetry.db with request_logs table and run migrations."""
    conn = sqlite3.connect("telemetry.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS request_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            prompt TEXT,
            status TEXT,
            similarity_score REAL,
            overlap_score REAL,
            latency_ms REAL,
            tokens_used INTEGER,
            tokens_saved INTEGER,
            antonym_blocked INTEGER DEFAULT 0,
            faiss_candidates_examined INTEGER DEFAULT 0,
            faiss_similarity REAL DEFAULT 0.0,
            retrieval_method TEXT DEFAULT 'faiss',
            faiss_retrieval_latency_ms REAL DEFAULT 0.0,
            cache_validation_latency_ms REAL DEFAULT 0.0,
            embedding_latency_ms REAL DEFAULT 0.0,
            faiss_search_latency_ms REAL DEFAULT 0.0
        )
    """)
    conn.commit()
    
    # SQLite Migration Logic: Add antonym_blocked column if it doesn't exist
    try:
        cursor.execute("SELECT antonym_blocked FROM request_logs LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating SQLite telemetry database: adding column 'antonym_blocked'...")
        cursor.execute("ALTER TABLE request_logs ADD COLUMN antonym_blocked INTEGER DEFAULT 0")
        conn.commit()
        
    # SQLite Migration Logic: Add new Phase 5 columns if they don't exist
    for col_name, col_type in [
        ("faiss_candidates_examined", "INTEGER DEFAULT 0"), 
        ("faiss_similarity", "REAL DEFAULT 0.0"), 
        ("retrieval_method", "TEXT DEFAULT 'faiss'"),
        ("faiss_retrieval_latency_ms", "REAL DEFAULT 0.0"),
        ("cache_validation_latency_ms", "REAL DEFAULT 0.0"),
        ("embedding_latency_ms", "REAL DEFAULT 0.0"),
        ("faiss_search_latency_ms", "REAL DEFAULT 0.0")
    ]:
        try:
            cursor.execute(f"SELECT {col_name} FROM request_logs LIMIT 1")
        except sqlite3.OperationalError:
            print(f"Migrating SQLite telemetry database: adding column '{col_name}'...")
            cursor.execute(f"ALTER TABLE request_logs ADD COLUMN {col_name} {col_type}")
            conn.commit()
            
    conn.close()

def log_request(prompt: str, status: str, similarity_score: float, overlap_score: float, 
                latency_ms: float, tokens_used: int, tokens_saved: int, antonym_blocked: int = 0,
                faiss_candidates_examined: int = 0, faiss_similarity: float = 0.0,
                retrieval_method: str = "faiss", faiss_retrieval_latency_ms: float = 0.0,
                cache_validation_latency_ms: float = 0.0, embedding_latency_ms: float = 0.0, 
                faiss_search_latency_ms: float = 0.0):
    """Insert request log entry into SQLite telemetry database."""
    try:
        conn = sqlite3.connect("telemetry.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO request_logs 
            (prompt, status, similarity_score, overlap_score, latency_ms, tokens_used, tokens_saved, antonym_blocked,
             faiss_candidates_examined, faiss_similarity, retrieval_method, faiss_retrieval_latency_ms, cache_validation_latency_ms,
             embedding_latency_ms, faiss_search_latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (prompt, status, similarity_score, overlap_score, latency_ms, tokens_used, tokens_saved, antonym_blocked,
              faiss_candidates_examined, faiss_similarity, retrieval_method, faiss_retrieval_latency_ms, cache_validation_latency_ms,
              embedding_latency_ms, faiss_search_latency_ms))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging telemetry to SQLite: {e}")

# Run database setup
init_db()

app = FastAPI(title="LexiCut FastAPI Service")

class ChatRequest(BaseModel):
    prompt: str = None
    query: str = None

@app.get("/")
def read_root():
    return {
        "message": "Welcome to LexiCut API",
        "libraries": {
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "google_antigravity": "available" if HAS_ANTIGRAVITY_SDK else "fallback mode"
        }
    }

@app.post("/chat")
async def chat(payload: ChatRequest):
    prompt = payload.prompt or payload.query
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt/Query cannot be empty")
    
    start_time = time.perf_counter()
    
    # 3. Check the semantic cache
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
    ) = cache_instance.check(prompt)
    
    # 4. IF IT IS A CACHE HIT:
    if cached_answer is not None:
        latency = (time.perf_counter() - start_time) * 1000.0
        tokens_saved = int(len(prompt.split()) * 1.3)
        
        log_request(
            prompt=prompt,
            status='HIT',
            similarity_score=similarity_score,
            overlap_score=overlap_score,
            latency_ms=latency,
            tokens_used=0,
            tokens_saved=tokens_saved,
            antonym_blocked=0,
            faiss_candidates_examined=candidates_examined,
            faiss_similarity=faiss_similarity,
            retrieval_method=retrieval_method,
            faiss_retrieval_latency_ms=embedding_latency_ms + faiss_search_latency_ms,
            cache_validation_latency_ms=cache_validation_latency_ms,
            embedding_latency_ms=embedding_latency_ms,
            faiss_search_latency_ms=faiss_search_latency_ms
        )
        
        return {
            "answer": cached_answer,
            "telemetry": {
                "source": "cache",
                "latency_ms": latency,
                "tokens_saved": tokens_saved
            }
        }
    
    # 5. IF IT IS A CACHE MISS:
    else:
        response_text = ""
        tokens_used = 0
        try:
            config = LocalAgentConfig(
                system_instructions="You are an expert assistant for codebase navigation."
            )
            async with Agent(config) as agent:
                response = await agent.chat(prompt)
                response_text = await response.text()
                
                # Capture actual tokens used from the SDK response
                if hasattr(response, 'usage_metadata'):
                    tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)
                elif hasattr(response, 'token_count'):
                    tokens_used = getattr(response, 'token_count', 0)
        except Exception as e:
            print(f"Error calling LLM Agent: {e}")
            response_text = f"Processed your query: '{prompt}'"
            
        latency = (time.perf_counter() - start_time) * 1000.0
        if tokens_used == 0:
            tokens_used = int((len(prompt.split()) + len(response_text.split())) * 1.3)
        
        # Save the new (prompt, response_text) pair into cache
        # Only cache if it was not blocked due to an antonym conflict to prevent bad caches
        if not antonym_blocked:
            cache_instance.add(prompt, response_text)
        
        log_request(
            prompt=prompt,
            status='MISS',
            similarity_score=similarity_score,
            overlap_score=overlap_score,
            latency_ms=latency,
            tokens_used=tokens_used,
            tokens_saved=0,
            antonym_blocked=1 if antonym_blocked else 0,
            faiss_candidates_examined=candidates_examined,
            faiss_similarity=faiss_similarity,
            retrieval_method=retrieval_method,
            faiss_retrieval_latency_ms=embedding_latency_ms + faiss_search_latency_ms,
            cache_validation_latency_ms=cache_validation_latency_ms,
            embedding_latency_ms=embedding_latency_ms,
            faiss_search_latency_ms=faiss_search_latency_ms
        )
        
        return {
            "answer": response_text,
            "telemetry": {
                "source": "llm",
                "latency_ms": latency,
                "tokens_used": tokens_used
            }
        }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
