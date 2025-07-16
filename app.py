from waitress import serve
from flask import Flask, request, jsonify
import faiss
import numpy as np
import fastembed
import os
import json

app = Flask(__name__)

# Load embedding model
model = fastembed.TextEmbedding('BAAI/bge-small-en-v1.5')
embedding_dim = len(list(model.embed(["test"]))[0]) 

# Paths
INDEX_FILE = "vector.index"
TEXTS_FILE = "texts.json"

# Load or initialize FAISS index
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    print("Loaded FAISS index from disk.")
else:
    index = faiss.IndexFlatL2(embedding_dim)
    print("Initialized empty FAISS index.")

# Load or initialize texts
if os.path.exists(TEXTS_FILE):
    with open(TEXTS_FILE, 'r') as f:
        texts = json.load(f)
    print(f"Loaded {len(texts)} text chunks.")
else:
    texts = []
    print("Initialized empty texts list.")

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
        return jsonify({'results': [], 'message': 'No data in index.'}), 200

    embedding = np.array(list(model.embed([query]))).astype('float32')  # ✅ FIXED

    try:
        D, I = index.search(embedding, min(k, index.ntotal))
        results = [texts[i] for i in I[0] if 0 <= i < len(texts)]
        return jsonify({'results': results})
    except Exception as e:
        print(f"SEARCH ERROR: {e}")
        return jsonify({'error': 'Vector search failed.', 'details': str(e)}), 500

    
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'OK',
        'vectors_in_index': index.ntotal,
        'texts_stored': len(texts)
    })



if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5001)
