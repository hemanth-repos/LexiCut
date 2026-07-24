import os
import faiss
import numpy as np

class FaissVectorIndex:
    def __init__(self, index_file: str = "faiss.index"):
        self.index_file = index_file
        self.dimension = None
        self.index = None
        # Try to load if index file exists
        if os.path.exists(self.index_file):
            try:
                self.load_index()
            except Exception as e:
                print(f"Error loading index on init from {self.index_file}: {e}")

    def _init_index(self, dimension: int):
        self.dimension = dimension
        # Use IndexFlatIP for Cosine similarity via inner product of normalized vectors
        base_index = faiss.IndexFlatIP(self.dimension)
        # Use IndexIDMap to map vector coordinates to integer IDs
        self.index = faiss.IndexIDMap(base_index)

    def add_vector(self, query_id: int, embedding: np.ndarray):
        # Convert embedding to numpy array of float32
        emb = np.array(embedding, dtype=np.float32)
        if len(emb.shape) == 1:
            emb = emb.reshape(1, -1)
        elif len(emb.shape) == 2:
            pass
        else:
            raise ValueError(f"Invalid embedding shape: {emb.shape}")
            
        dim = emb.shape[1]
        if self.index is None:
            self._init_index(dim)
        elif dim != self.dimension:
            raise ValueError(f"Embedding dimension {dim} does not match index dimension {self.dimension}")
            
        # Normalize the embedding for cosine similarity via inner product
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_normalized = emb / norms
        
        # Remove any existing entry for this ID to avoid duplicate mappings
        ids = np.array([query_id], dtype=np.int64)
        try:
            self.index.remove_ids(ids)
        except Exception as e:
            # If not found or unsupported, ignore
            pass
            
        # Add to the index
        self.index.add_with_ids(emb_normalized, ids)

    def search(self, embedding: np.ndarray, top_k: int = 5):
        if self.index is None or self.index.ntotal == 0:
            return [], []
            
        emb = np.array(embedding, dtype=np.float32)
        if len(emb.shape) == 1:
            emb = emb.reshape(1, -1)
        elif len(emb.shape) == 2:
            pass
        else:
            raise ValueError(f"Invalid embedding shape: {emb.shape}")
            
        dim = emb.shape[1]
        if dim != self.dimension:
            raise ValueError(f"Query embedding dimension {dim} does not match index dimension {self.dimension}")
            
        # Normalize the query embedding
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_normalized = emb / norms
        
        # Determine actual k (cannot search for more than index.ntotal)
        actual_k = min(top_k, self.index.ntotal)
        if actual_k <= 0:
            return [], []
            
        distances, ids = self.index.search(emb_normalized, actual_k)
        
        # Since we query with one vector, return the lists for that vector
        return distances[0].tolist(), ids[0].tolist()

    def save_index(self, filepath: str = None):
        path = filepath or self.index_file
        if self.index is not None:
            faiss.write_index(self.index, path)
            print(f"Saved FAISS index to {path}")
        else:
            print("No index to save.")

    def load_index(self, filepath: str = None):
        path = filepath or self.index_file
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            self.dimension = self.index.d
            print(f"Loaded FAISS index from {path} with dimension {self.dimension}")
        else:
            print(f"Index file {path} not found for loading.")
