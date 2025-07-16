from waitress import serve
from flask import Flask, request, jsonify
import faiss
import numpy as np
import fastembed
import os
import sqlite3
import json

app = Flask(__name__)

model = fastembed.TextEmbedding('BAAI/bge-small-en-v1.5')
embedding_dim = len(list(model.embed(["test"]))[0])

# Paths
INDEX_FILE = "vector.index"
DB_FILE = "chunks.db"

# Load or create FAISS index
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(embedding_dim)

# SQLite setup
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL
)
''')
conn.commit()

def store_texts(chunks):
    cursor.executemany('INSERT INTO chunks (text) VALUES (?)', [(chunk,) for chunk in chunks])
    conn.commit()
    ids = cursor.execute('SELECT last_insert_rowid()').fetchone()[0]
    start_id = ids - len(chunks) + 1
    return list(range(start_id, ids + 1))

def fetch_texts_by_ids(ids):
    placeholders = ','.join(['?'] * len(ids))
    cursor.execute(f'SELECT text FROM chunks WHERE id IN ({placeholders})', ids)
    return [row[0] for row in cursor.fetchall()]

@app.route('/add_chunks', methods=['POST'])
def add_chunks():
    data = request.json
    chunks = data.get('chunks', [])

    if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
        return jsonify({'error': 'Invalid input. Expected list of strings.'}), 400

    # Generate embeddings
    embeddings = list(model.embed(chunks))
    index.add(np.array(embeddings).astype('float32'))

    # Store chunks in SQLite
    ids = store_texts(chunks)

    # Save index after batch
    faiss.write_index(index, INDEX_FILE)

    return jsonify({'status': 'chunks added', 'added_count': len(ids)})

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    k = int(request.json.get('top_k', 3))

    if not query.strip():
        return jsonify({'error': 'Empty query string.'}), 400

    if index.ntotal == 0:
        return jsonify({'results': [], 'message': 'No data available in index.'}), 200

    embedding = np.array(list(model.embed([query]))).astype('float32')
    D, I = index.search(embedding, min(k, index.ntotal))

    ids = [int(idx) + 1 for idx in I[0]]  # SQLite rowid starts from 1
    results = fetch_texts_by_ids(ids)

    return jsonify({'results': results})

@app.route('/health', methods=['GET'])
def health():
    total_chunks = cursor.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
    return jsonify({'status': 'OK', 'vectors_in_index': index.ntotal, 'texts_stored': total_chunks})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5001)
