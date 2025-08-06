import os
import sqlite3
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from fastembed import TextEmbedding
from contextlib import contextmanager
import pdfplumber
from starlette.responses import JSONResponse
import tempfile
import psutil
import time
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
DB_FILE = os.getenv("DB_FILE", "chunks.db")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 500))
PORT = int(os.getenv("PORT", 8080))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
SUB_BATCH_SIZE = int(os.getenv("SUB_BATCH_SIZE", 10))
MAX_PAYLOAD_SIZE = int(os.getenv("MAX_PAYLOAD_SIZE", 300000))

# Initialize embedding model
model = TextEmbedding(EMBEDDING_MODEL)

# Initialize ChromaDB in-memory
client = chromadb.Client()
collection = client.get_or_create_collection(name="chunks")

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

def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of approximately chunk_size characters."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def log_memory_usage(stage: str):
    """Log current memory usage in MiB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_usage = mem_info.rss / 1024 / 1024
    logger.info(f"Memory usage at {stage}: {mem_usage:.2f} MiB")

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    start_time = time.time()
    log_memory_usage("start of PDF processing")

    try:
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        log_memory_usage("after PDF upload")

        # Extract text from PDF
        chunks = []
        with pdfplumber.open(tmp_file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    page_chunks = split_text_into_chunks(text, CHUNK_SIZE)
                    chunks.extend(page_chunks)

        # Delete the temporary PDF file
        os.remove(tmp_file_path)
        logger.info(f"Deleted temporary PDF file: {tmp_file_path}")
        log_memory_usage("after PDF text extraction")

        if not chunks:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")

        if len(chunks) > MAX_BATCH_SIZE:
            raise HTTPException(status_code=400, detail=f"Number of chunks ({len(chunks)}) exceeds maximum of {MAX_BATCH_SIZE}")

        # Generate embeddings in smaller batches
        embeddings = []
        for i in range(0, len(chunks), SUB_BATCH_SIZE):
            batch = chunks[i:i + SUB_BATCH_SIZE]
            batch_embeddings = list(model.embed(batch))
            embeddings.extend(batch_embeddings)
            log_memory_usage(f"after embedding batch {i//SUB_BATCH_SIZE + 1}")
            batch_embeddings = None

        # Add to ChromaDB
        ids = store_texts(chunks)
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[str(id) for id in ids]
        )
        log_memory_usage("after ChromaDB insert")

        processing_time = time.time() - start_time
        logger.info(f"Processed PDF and added {len(ids)} chunks in {processing_time:.2f} seconds")
        return {"status": "pdf processed", "added_count": len(ids)}
    except Exception as e:
        if 'tmp_file_path' in locals():
            try:
                os.remove(tmp_file_path)
                logger.info(f"Deleted temporary PDF file due to error: {tmp_file_path}")
            except:
                pass
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_chunks")
async def add_chunks(data: AddChunksRequest, request: Request):
    content_length = int(request.headers.get("content-length", 0))
    if content_length > MAX_PAYLOAD_SIZE:
        raise HTTPException(status_code=400, detail=f"Payload size ({content_length} bytes) exceeds maximum of {MAX_PAYLOAD_SIZE} bytes")

    chunks = data.chunks
    if not chunks:
        raise HTTPException(status_code=400, detail="Empty chunk list")
    
    if len(chunks) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")

    start_time = time.time()
    log_memory_usage("start of add_chunks")

    try:
        embeddings = []
        for i in range(0, len(chunks), SUB_BATCH_SIZE):
            batch = chunks[i:i + SUB_BATCH_SIZE]
            batch_embeddings = list(model.embed(batch))
            embeddings.extend(batch_embeddings)
            log_memory_usage(f"after embedding batch {i//SUB_BATCH_SIZE + 1}")
            batch_embeddings = None
        
        # Add to ChromaDB
        ids = store_texts(chunks)
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[str(id) for id in ids]
        )
        log_memory_usage("after ChromaDB insert")

        processing_time = time.time() - start_time
        logger.info(f"Added {len(ids)} chunks to index and database in {processing_time:.2f} seconds")
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

    try:
        embedding = list(model.embed([query]))[0]
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, collection.count())
        )
        ids = [int(id) for id in results["ids"][0]]
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
            "vectors_in_index": collection.count(),
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
