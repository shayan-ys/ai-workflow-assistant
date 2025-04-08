import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging
from typing import List

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(
        self,
        chunk_char_size: int = 1250,
        num_chunks: int = 10,
        chunk_overlap: int = 200,
        embedding_name: str = "BAAI/bge-large-en-v1.5",
        db_path: str = "./chromadb",
    ):
        self.chunk_char_size = chunk_char_size
        self.chunk_overlap = chunk_overlap
        self.num_chunks = num_chunks
        self.embedding_name = embedding_name
        self.collection_name = f"rag_collection_{self.get_rag_id()}"
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_name
        )

        logger.info("Initializing ChromaDB Client...")
        self.chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )
        logger.info("ChromaDB Client initialized successfully.")

    def get_rag_id(self):
        rag_id = f"{self.num_chunks}chunks-{self.chunk_char_size}_emb-{self.embedding_name.replace('/', '_')}"  # noqa
        return rag_id.replace("-", "_")

    def list_collections(self):
        return self.chroma_client.list_collections()

    def collection_exists(self):
        return self.collection_name in [c.name for c in self.list_collections()]

    def query_collection(self, query: str) -> List[str]:
        if not self.collection_exists():
            logger.error(f"Collection {self.collection_name} does not exist.")
            return []

        collection = self.chroma_client.get_collection(
            name=self.collection_name, embedding_function=self.emb_fn
        )

        results = collection.query(query_texts=[query], n_results=self.num_chunks)
        return results.get("documents", [[]])[0]
