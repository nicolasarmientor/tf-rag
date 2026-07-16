
import sys
from chunker import Chunk, chunk_documents
from loader import load_documents
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    return embeddings

if __name__ == "__main__":
    sys.path.append("..")

    documents = load_documents()
    chunks = chunk_documents(documents)
    embeddings = embed_chunks(chunks)

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimensionality: {len(embeddings[0])}")