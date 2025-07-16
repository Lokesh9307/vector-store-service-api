from waitress import serve
from flask import Flask, request, jsonify
import faiss
import numpy as np
import fastembed
import os
import json

app = Flask(__name__)

model = fastembed.TextEmbedding('BAAI/bge-small-en-v1.5')

INDEX_FILE = "vector.index"
TEXTS_FILE = "texts.json"

# Load or initialize FAISS index
embedding_dim = 384  # Adjust if using a different model
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(embedding_dim)

# Load or initialize stored texts
if os.path.exists(TEXTS_FILE):
    with open(TEXTS_FILE, 'r') as f:
        texts = json.load(f)
else:
    texts = []

@app.route('/add_chunks', methods=['POST'])
def add_chunks():
    data = request.json
    chunks = data.get('chunks', [])

    if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
        return jsonify({'error': 'Invalid input. Expected list of strings.'}), 400

    embeddings = list(model.embed(chunks))

    index.add(np.array(embeddings).astype('float32'))

    texts.extend(chunks)

    faiss.write_index(index, INDEX_FILE)
    with open(TEXTS_FILE, 'w') as f:
        json.dump(texts, f)

    return jsonify({'status': 'chunks added', 'total_chunks': len(texts)})

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    k = int(request.json.get('top_k', 3))

    if not query.strip():
        return jsonify({'error': 'Empty query string.'}), 400

    if index.ntotal == 0 or len(texts) == 0:
        return jsonify({'results': [], 'message': 'No data available in index.'})

    embedding = np.array(model.embed([query])).astype('float32')
    D, I = index.search(embedding, min(k, index.ntotal))

    results = [texts[i] for i in I[0] if 0 <= i < len(texts)]

    return jsonify({'results': results})


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5001)
