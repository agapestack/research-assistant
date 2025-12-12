from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import hashlib


class VectorStore:
    """Qdrant vector store for document embeddings."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "papers",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(embedding_model)
        self.vector_size = self.encoder.get_sentence_embedding_dimension()
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def _generate_id(self, doc: Document) -> str:
        key = f"{doc.metadata['source']}_{doc.metadata['page']}_{doc.metadata['chunk']}"
        return hashlib.md5(key.encode()).hexdigest()

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return

        points = []
        for doc in documents:
            vector = self.encoder.encode(doc.page_content).tolist()
            payload = {"content": doc.page_content, **doc.metadata}
            points.append(PointStruct(
                id=self._generate_id(doc),
                vector=vector,
                payload=payload,
            ))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for similar documents."""
        query_vector = self.encoder.encode(query).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
        ).points

        return [
            {
                "content": hit.payload.pop("content"),
                "metadata": hit.payload,
                "score": hit.score,
            }
            for hit in results
        ]

    def count(self) -> int:
        """Return number of documents in store."""
        return self.client.count(collection_name=self.collection_name).count

    def clear(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
