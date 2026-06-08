"""
Embedding + retrieval for The Unofficial Guide (Milestone 4).

Pipeline stages 3-4 from planning.md / the architecture diagram:
    chunks.json  ->  embed with all-MiniLM-L6-v2  ->  store in ChromaDB
    query        ->  embed  ->  top-k semantic search  ->  ranked chunks

Run once to build the index and smoke-test retrieval:
    python search.py
Then other code (Milestone 5) imports `retrieve`:
    from search import retrieve
"""

import json
import os

import chromadb
from chromadb.utils import embedding_functions

# --- config (from planning.md Retrieval Approach) --------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(ROOT, "data", "chunks.json")
CHROMA_DIR = os.path.join(ROOT, "chroma_db")     # persisted on disk (gitignored)
COLLECTION_NAME = "unofficial_guide"
EMBED_MODEL = "all-MiniLM-L6-v2"                  # local, free, built for short text
TOP_K = 5


# ---------------------------------------------------------------------------
# ChromaDB collection
# ---------------------------------------------------------------------------
# PersistentClient writes the vector store to disk so we embed once and reuse it
# across runs (an in-memory Client() would re-embed every time).
#
# The embedding_function tells Chroma how to turn text into vectors. We hand it
# all-MiniLM-L6-v2 so the SAME model embeds both the stored chunks and the query
# at search time -- if they used different models, the vectors wouldn't be
# comparable and retrieval would be meaningless.
#
# metadata={"hnsw:space": "cosine"} sets the similarity metric to cosine
# distance (0 = identical, higher = less similar). all-MiniLM vectors are
# direction-based, so cosine is the right metric -- and it makes the
# "distance < 0.5 = good match" rule of thumb meaningful.
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Build the index: load chunks, embed, store with metadata
# ---------------------------------------------------------------------------
def _clean_metadata(meta):
    """Chroma metadata values must be str/int/float/bool -- no None."""
    out = {}
    for k, v in meta.items():
        out[k] = "" if v is None else v
    return out


def build_index():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    collection = get_collection()

    ids = [f"{c['metadata']['source']}_{c['metadata']['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [_clean_metadata(c["metadata"]) for c in chunks]

    # upsert = insert or overwrite by id, so re-running is safe (no duplicates).
    # Chroma calls the embedding_function on `documents` for us here.
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}' "
          f"(model: {EMBED_MODEL}, metric: cosine)")
    return collection


# ---------------------------------------------------------------------------
# Retrieve: embed the query and return the top-k most similar chunks
# ---------------------------------------------------------------------------
def retrieve(query, k=TOP_K):
    """Return the k most relevant chunks as a list of dicts:
    {text, metadata, distance}  (distance: lower = more relevant)."""
    collection = get_collection()
    res = collection.query(
        query_texts=[query],          # Chroma embeds this with the same model
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma returns parallel lists wrapped in an outer list (one per query).
    results = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        results.append({"text": doc, "metadata": meta, "distance": dist})
    return results


# ---------------------------------------------------------------------------
# main: build index, then smoke-test retrieval on a few eval questions
# ---------------------------------------------------------------------------
def main():
    build_index()

    # 3 of the 5 questions from the Evaluation Plan in planning.md (verbatim),
    # so this smoke test exercises the real eval queries.
    test_queries = [
        "What do students say about Amit Patel's attendance policy?",
        "Is Fraida Fund's course a heavy workload, and who is it suited for?",
        "What are the main complaints about Chinmay Hegde's grading?",
    ]
    for q in test_queries:
        print("\n" + "=" * 70)
        print("QUERY:", q)
        print("=" * 70)
        for i, r in enumerate(retrieve(q, k=TOP_K), 1):
            m = r["metadata"]
            tag = m.get("professor") or m.get("type")
            print(f"\n  {i}. distance={r['distance']:.3f}  [{m['source']}]  ({tag})")
            print(f"     {r['text'][:200]}")

if __name__ == "__main__":
    main()
