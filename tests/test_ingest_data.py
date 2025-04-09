from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from swayai.rag_engine import RAGEngine
from typing import List

logger = logging.getLogger(__name__)


def create_collection_from_json_docs(
    rag_engine: RAGEngine, project_jsons: List[dict], force=False
):
    if rag_engine.collection_exists() and not force:
        logger.info(
            f"Collection {rag_engine.collection_name} already exists. Skipping creation"
        )
        return

    collection = rag_engine.chroma_client.create_collection(
        name=rag_engine.collection_name,
        embedding_function=rag_engine.emb_fn,
        get_or_create=True,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_engine.chunk_char_size, chunk_overlap=rag_engine.chunk_overlap
    )

    for idx, doc in enumerate(project_jsons):
        full_text = "\n".join([f"{k}: {v}" for k, v in doc.items()])
        chunks = text_splitter.split_text(full_text)
        if not chunks:
            continue
        ids = [f"proj_{idx}_chunk_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
