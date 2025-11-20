import os
import json
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer, util

# Optional: try to import faiss if available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("[WARN] faiss not installed — falling back to cosine similarity.")
    FAISS_AVAILABLE = False


# --- Resolve absolute paths dynamically ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "learnora_finetuned_stv2")
METADATA_PATH = os.path.join(BASE_DIR, "datasets", "learnora_metadata_final.json")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "datasets", "learnora_faiss_final.index")

# --- Globals ---
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_dataset: list[dict[str, Any]] | None = None
_use_faiss: bool = False


# --- Load model ---
def _load_model() -> SentenceTransformer:
    global _model, _use_faiss
    if _model is None:
        if os.path.exists(MODEL_PATH):
            print(f"[INFO] Loading fine-tuned model from: {MODEL_PATH}")
            _model = SentenceTransformer(MODEL_PATH)
        else:
            print(f"[WARN] Fine-tuned model not found. Using pretrained model instead.")
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


# --- Load FAISS index ---
def _load_index() -> "faiss.Index | None":
    global _index, _use_faiss
    if not FAISS_AVAILABLE:
        return None
    if _index is None:
        if os.path.exists(FAISS_INDEX_PATH):
            print(f"[INFO] Loading FAISS index from: {FAISS_INDEX_PATH}")
            _index = faiss.read_index(FAISS_INDEX_PATH)
            _use_faiss = True
        else:
            print(f"[WARN] No FAISS index found. Will use cosine similarity.")
            _use_faiss = False
    return _index


# --- Load dataset ---
def _load_dataset() -> list[dict[str, Any]]:
    global _dataset
    if _dataset is None:
        print(f"[INFO] Loading metadata from: {METADATA_PATH}")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _dataset = json.load(f)
    return _dataset


# --- Semantic search ---
def semantic_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    model = _load_model()
    data = _load_dataset()
    index = _load_index()

    # Encode query
    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    results: List[Dict[str, Any]] = []

    if _use_faiss and index is not None:
        # ---- Use FAISS search ----
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)
        for idx, score in zip(indices[0], distances[0]):
            if 0 <= idx < len(data):
                item = dict(data[idx])
                item["similarity_score"] = float(score)
                results.append(item)
    else:
        # ---- Use cosine similarity ----
        print("[INFO] Using cosine similarity search...")
        texts = [item["title"] + " " + item.get("summary", "") for item in data]
        corpus_embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        scores = np.dot(corpus_embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]

        for idx in top_indices:
            item = dict(data[idx])
            item["similarity_score"] = float(scores[idx])
            results.append(item)

    return results
