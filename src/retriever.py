import numpy as np
import json
import yaml
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class DocumentRetriever:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = SentenceTransformer(self.config["embedder"]["model_name"])
        self.top_k = self.config["retriever"]["top_k"]
        self.similarity_threshold = self.config["retriever"]["similarity_threshold"]

        self.embeddings = None
        self.index = None
        self.documents = None

    def load_embeddings(self):
        embeddings_path = self.config["data"]["embeddings_path"]
        index_path = self.config["data"]["index_path"]

        self.embeddings = np.load(embeddings_path)

        with open(index_path, "r") as f:
            self.index = json.load(f)

    def load_documents(self):
        documents = {}
        docs_path = self.config["data"]["documents_path"]

        for item in self.index:
            filepath = f"{docs_path}/{item['source']}"
            with open(filepath, "r", encoding="utf-8") as f:
                documents[item["id"]] = f.read()

        self.documents = documents

    def retrieve(self, query: str) -> List[Dict[str, any]]:
        if self.embeddings is None:
            self.load_embeddings()
            self.load_documents()

        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

        similarities = np.dot(self.embeddings, query_embedding)
        similarities = similarities / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[-self.top_k :][::-1]

        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= self.similarity_threshold:
                doc_id = self.index[idx]["id"]
                results.append(
                    {
                        "id": doc_id,
                        "content": self.documents[doc_id],
                        "similarity": similarity,
                        "source": self.index[idx]["source"],
                    }
                )

        return results
