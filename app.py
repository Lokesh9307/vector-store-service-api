from waitress import serve
from flask import Flask, request, jsonify
import faiss
import numpy as np
import fastembed
from itertools import chain

app = Flask(__name__)

model = fastembed.TextEmbedding('BAAI/bge-small-en-v1.5')

vectors = []
texts = []

def flatten_chunks(chunks):
    # Flatten list of lists into flat list of strings
    if chunks and isinstance(chunks[0], list):
        return list(chain.from_iterable(chunks))
    return chunks

@app.route('/add_chunks', methods=['POST'])
def add_chunks():
    global vectors, texts
    data = request.json
    chunks = data.get('chunks', [])

    # Flatten chunks input if necessary
    chunks = flatten_chunks(chunks)

    # Validate input
    if not all(isinstance(chunk, str) for chunk in chunks):
        return jsonify({'error': 'Invalid input. Expected list of strings.'}), 400

    embeddings = list(model.embed(chunks))

    vectors.extend(embeddings)
    texts.extend(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(vectors).astype('float32'))

    faiss.write_index(index, "vector.index")

    return jsonify({'status': 'chunks added'})

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    k = int(request.json.get('top_k', 3))

    embedding = list(model.embed([query]))

    index = faiss.read_index("vector.index")
    D, I = index.search(np.array(embedding).astype('float32'), k)

    results = [texts[i] for i in I[0]]
    return jsonify({'results': results})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5001)
