import pytest
import numpy as np
from src.retriever import DocumentRetriever
import json


@pytest.fixture
def retriever():
    return DocumentRetriever()


@pytest.fixture
def setup_test_embeddings(tmp_path):
    embeddings = np.random.rand(3, 384)
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    index = [
        {"id": "doc1", "source": "doc1.txt"},
        {"id": "doc2", "source": "doc2.txt"},
        {"id": "doc3", "source": "doc3.txt"},
    ]
    index_path = tmp_path / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f)

    return embeddings_path, index_path


def test_retriever_initialization(retriever):
    assert retriever.model is not None
    assert retriever.top_k > 0


def test_retrieve_returns_correct_number_of_docs(retriever):
    if retriever.embeddings is None:
        pytest.skip("Embeddings not loaded")

    results = retriever.retrieve("test query")
    assert len(results) <= retriever.top_k
