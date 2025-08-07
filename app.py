import os
import sqlite3
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import psutil
import time
from contextlib import contextmanager
from fastembed import TextEmbedding
import pdfplumber
import chromadb

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware, allow_origins=["*","https://pdfchatbot-api-117429664165.europe-west1.run.app"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Config
DB_FILE = os.getenv("DB_FILE", "chunks.db")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 500))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
SUB_BATCH_SIZE = int(os.getenv("SUB_BATCH_SIZE", 10))
MAX_PAYLOAD_SIZE = int(os.getenv("MAX_PAYLOAD_SIZE", 300000))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Embedding model
model = TextEmbedding(EMBEDDING_MODEL)

# ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection(name="chunks")

# SQLite setup
conn = sqlite3.connect(DB_FILE, check_same_thread=False, isolation_level=None)
conn.execute("PRAGMA journal_mode=WAL")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL
)""")
conn.commit()

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
        return list(range(last_id - len(chunks) + 1, last_id + 1))

def fetch_texts_by_ids(ids: List[int]) -> List[str]:
    with get_db_cursor() as cur:
        placeholders = ','.join('?' * len(ids))
        cur.execute(f"SELECT text FROM chunks WHERE id IN ({placeholders})", ids)
        return [row[0] for row in cur.fetchall()]

def log_memory_usage(stage: str):
    mem = psutil.Process().memory_info().rss / (1024 ** 2)
    logger.info(f"Memory usage at {stage}: {mem:.2f} MiB")

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings = []
    for i in range(0, len(chunks), SUB_BATCH_SIZE):
        batch = chunks[i:i + SUB_BATCH_SIZE]
        batch_embeddings = model.embed(batch)
        embeddings.extend([e.tolist() for e in batch_embeddings])
        log_memory_usage(f"after embedding batch {i//SUB_BATCH_SIZE + 1}")
    return embeddings

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    start_time = time.time()
    log_memory_usage("start of PDF processing")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        log_memory_usage("after PDF upload")

        chunks = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    for i in range(0, len(text), CHUNK_SIZE):
                        chunks.append(text[i:i + CHUNK_SIZE])

        os.remove(tmp_path)
        logger.info(f"Deleted temporary PDF: {tmp_path}")
        log_memory_usage("after PDF text extraction")

        if not chunks:
            raise HTTPException(400, "No text extracted")
        if len(chunks) > MAX_BATCH_SIZE:
            raise HTTPException(400, f"Too many chunks: {len(chunks)} > {MAX_BATCH_SIZE}")

        embeddings = embed_chunks(chunks)
        ids = store_texts(chunks)
        collection.add(documents=chunks, embeddings=embeddings, ids=[str(i) for i in ids])
        log_memory_usage("after ChromaDB insert")

        logger.info(f"Processed PDF, added {len(ids)} chunks in {time.time() - start_time:.2f}s")
        return {"status": "pdf processed", "added_count": len(ids)}

    except Exception as e:
        if 'tmp_path' in locals():
            try:
                os.remove(tmp_path)
                logger.info(f"Removed temp file due to error: {tmp_path}")
            except:
                pass
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(500, str(e))

@app.post("/add_chunks")
async def add_chunks(data: AddChunksRequest, request: Request):
    size = int(request.headers.get("content-length", 0))
    if size > MAX_PAYLOAD_SIZE:
        raise HTTPException(400, f"Payload too large: {size} > {MAX_PAYLOAD_SIZE}")
    chunks = data.chunks or []
    if not chunks:
        raise HTTPException(400, "Empty chunk list")
    if len(chunks) > MAX_BATCH_SIZE:
        raise HTTPException(400, f"Too many chunks: {len(chunks)} > {MAX_BATCH_SIZE}")

    start_time = time.time()
    log_memory_usage("start of add_chunks")
    try:
        embeddings = embed_chunks(chunks)
        ids = store_texts(chunks)
        collection.add(documents=chunks, embeddings=embeddings, ids=[str(i) for i in ids])
        log_memory_usage("after ChromaDB insert")

        logger.info(f"Added {len(ids)} chunks in {time.time() - start_time:.2f}s")
        return {"status": "chunks added", "added_count": len(ids)}
    except Exception as e:
        logger.error(f"Error adding chunks: {e}")
        raise HTTPException(500, str(e))

@app.post("/search")
async def search(data: SearchRequest):
    query = data.query.strip()
    if not query:
        raise HTTPException(400, "Empty query")

    try:
        # Embed the query and convert to native list
        raw_embedding = list(model.embed([query]))[0]
        embedding = raw_embedding.tolist()  # Convert ndarray to Python list

        # Perform search with properly formatted embedding
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(data.top_k, collection.count())
        )

        # Convert IDs to int list
        ids = [int(i) for i in list(results["ids"][0])]
        texts = fetch_texts_by_ids(ids)

        logger.info(f"Search completed: {len(texts)} hits")
        return {"results": texts}

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    try:
        with get_db_cursor() as cur:
            total = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"status": "OK", "vectors_in_index": collection.count(), "texts_stored": total}
    except Exception as e:
        logger.error(f"Health error: {e}")
        raise HTTPException(500, str(e))

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Received {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Completed {request.method} {request.url.path}")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, "0.0.0.0", port=int(os.getenv("PORT", 8080)))
