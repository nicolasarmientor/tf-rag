import chromadb
import sys
from src.ingest.loader import load_documents
from src.ingest.chunker import Chunk, chunk_documents
from src.ingest.embed import embed_chunks, model
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "chroma_data"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(name="tf_guide")

def add_chunks(chunks: list[Chunk], embeddings) -> None:
    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {"source_path": chunk.source_path, "title": chunk.title}
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

def query_chunks(query_embedding, top_k: int = 5):
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k
    )

    return results

if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)
    embeddings = embed_chunks(chunks)

    add_chunks(chunks, embeddings)
    print(f"Items in collection after add: {collection.count()}")

    test_query = "What is a Keras sequential model?"
    query_embedding = model.encode(test_query)
    results = query_chunks(query_embedding, top_k=3)

    print(f"Query: {test_query}")
    
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"Source: {meta['source_path']} | Title: {meta['title']}")
        print(f"Text preview: {doc[:150]}")
        print("---")