import pytest
import numpy as np
from src.embedder import DocumentEmbedder


@pytest.fixture
def embedder():
    return DocumentEmbedder()


@pytest.fixture
def sample_documents():
    return [
        {
            "id": "doc1",
            "content": "This is a test document about returns.",
            "source": "test1.txt",
        },
        {
            "id": "doc2",
            "content": "This is a test document about shipping.",
            "source": "test2.txt",
        },
    ]


def test_embedder_initialization(embedder):
    assert embedder.model is not None
    assert embedder.batch_size > 0


def test_embed_documents(embedder, sample_documents):
    embeddings = embedder.embed_documents(sample_documents)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(sample_documents)
    assert embeddings.shape[1] > 0


def test_embeddings_are_normalized(embedder, sample_documents):
    embeddings = embedder.embed_documents(sample_documents)
    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0, atol=0.1)
