import os
import sqlite3
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
from fastembed import TextEmbedding
from contextlib import contextmanager
from starlette.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Environment variables
INDEX_FILE = os.getenv("INDEX_FILE", "vector.index")
DB_FILE = os.getenv("DB_FILE", "chunks.db")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 100))
PORT = int(os.getenv("PORT", 8080))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Initialize embedding model
model = TextEmbedding(EMBEDDING_MODEL)
embedding_dim = len(list(model.embed(["test"]))[0])

# SQLite connection pool
conn = sqlite3.connect(DB_FILE, check_same_thread=False, isolation_level=None)
conn.execute("PRAGMA journal_mode=WAL")
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL
)
''')
conn.commit()

# FAISS index
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(embedding_dim)

# SQLite context manager
@contextmanager
def get_db_cursor():
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise

class AddChunksRequest(BaseModel):
    chunks: List[str]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

def store_texts(chunks: List[str]) -> List[int]:
    with get_db_cursor() as cur:
        cur.executemany('INSERT INTO chunks (text) VALUES (?)', [(chunk,) for chunk in chunks])
        last_id = cur.execute('SELECT last_insert_rowid()').fetchone()[0]
        start_id = last_id - len(chunks) + 1
        return list(range(start_id, last_id + 1))

def fetch_texts_by_ids(ids: List[int]) -> List[str]:
    with get_db_cursor() as cur:
        placeholders = ','.join(['?'] * len(ids))
        cur.execute(f'SELECT text FROM chunks WHERE id IN ({placeholders})', ids)
        return [row[0] for row in cur.fetchall()]

@app.post("/add_chunks")
async def add_chunks(data: AddChunksRequest):
    chunks = data.chunks
    if not chunks:
        raise HTTPException(status_code=400, detail="Empty chunk list")
    
    if len(chunks) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")

    try:
        # Generate embeddings in smaller batches
        batch_size = 50
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings.extend(model.embed(batch))
        
        # Add to FAISS index
        index.add(np.array(embeddings).astype('float32'))

        # Store chunks in SQLite
        ids = store_texts(chunks)

        # Save FAISS index
        faiss.write_index(index, INDEX_FILE)

        logger.info(f"Added {len(ids)} chunks to index and database")
        return {"status": "chunks added", "added_count": len(ids)}
    except Exception as e:
        logger.error(f"Error adding chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(data: SearchRequest):
    query = data.query
    top_k = data.top_k

    if not query.strip():
        raise HTTPException(status_code=400, detail="Empty query string")

    if index.ntotal == 0:
        return {"results": [], "message": "No data available in index"}

    try:
        embedding = np.array(list(model.embed([query]))).astype('float32')
        D, I = index.search(embedding, min(top_k, index.ntotal))
        ids = [int(idx) + 1 for idx in I[0]]
        results = fetch_texts_by_ids(ids)

        logger.info(f"Search completed for query with {len(results)} results")
        return {"results": results}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    try:
        with get_db_cursor() as cur:
            total_chunks = cur.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        return {
            "status": "OK",
            "vectors_in_index": index.ntotal,
            "texts_stored": total_chunks
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Custom middleware to log request start and end
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Received {request.method} request to {request.url.path}")
    response = await call_next(request)
    logger.info(f"Completed {request.method} request to {request.url.path}")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)