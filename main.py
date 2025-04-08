from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import pandas as pd
from pydantic import BaseModel
from rag.data_science import infer_column_types
from rag.rag_engine import RAGEngine
import tempfile
from tests.test_rag_engine import test_rag
from typing import List

logger = logging.getLogger(__name__)

app = FastAPI()
rag_engine = RAGEngine(num_chunks=5)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryResult(BaseModel):
    chunks: List[str]


@app.get("/tests/rag")
def ingest():
    test_rag(rag_engine)
    return {"test complete": True}


@app.post("/query", response_model=QueryResult)
async def query_with_csv(csv_file: UploadFile = File(...)):
    # Save and read the uploaded CSV
    temp_path = tempfile.mkstemp(suffix=".csv")[1]
    with open(temp_path, "wb") as f:
        content = await csv_file.read()
        f.write(content)

    df = pd.read_csv(temp_path)
    os.remove(temp_path)

    # Extract column names for the query string
    # column_names = df.columns.tolist()
    # query_str = "Customer data columns: " + ", ".join(column_names)
    # Extract column name and type information for the query string
    col_types = infer_column_types(df)
    query_str = "Customer data columns:\n" + "\n".join(
        [f"- {col}: {dtype}" for col, dtype in col_types.items()]
    )
    logger.debug(f"Query string: {query_str}")

    # Run query through RAG
    chunks = rag_engine.query_collection(query_str)

    return {"chunks": chunks}
