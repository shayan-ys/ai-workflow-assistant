from rag.rag_engine import RAGEngine
from tests.test_ingest_data import create_collection_from_json_docs

sample_projects = [
    {
        "project": "Loan Default Prediction",
        "data_summary": {
            "columns": ["loan_id", "credit_score", "default_flag"],
            "types": ["string", "float", "int"]
        },
        "use_case": "Predict loan default",
        "why_it_fits": "Has features + label",
        "suggested_nodes": ["DataIngest", "Impute", "TrainClassifier"]
    }
]


def test_rag(engine: RAGEngine = None):
    engine = engine or RAGEngine()
    create_collection_from_json_docs(engine, sample_projects, force=True)

    query = "I want to predict who might default on a loan"
    results = engine.query_collection(query)

    print("🔍 Retrieved Chunks:")
    for chunk in results:
        print("-" * 40)
        print(chunk)
