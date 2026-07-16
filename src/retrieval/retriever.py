
from src.ingest.embed import model
from src.retrieval.vector_store import query_chunks

DISTANCE_THRESHOLD = 1.3

def retrieve(question: str, top_k: int = 5):
    query_embedding = model.encode(question)
    results = query_chunks(query_embedding, top_k=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        if distance <= DISTANCE_THRESHOLD:
            retrieved.append({
                "text": document,
                "source_path": metadata["source_path"],
                "title": metadata["title"],
                "distance": distance
            })

    return retrieved

if __name__ == "__main__":
    question = "What is a Keras sequential model?"
    results = retrieve(question, top_k=3)

    for r in results:
        print(f"[{r['distance']:.3f}] {r['title']} ({r['source_path']})")