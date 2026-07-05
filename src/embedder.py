import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from typing import List, Dict
import yaml


class DocumentEmbedder:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = SentenceTransformer(self.config["embedder"]["model_name"])
        self.batch_size = self.config["embedder"]["batch_size"]
        self.max_length = self.config["embedder"]["max_length"]

    def load_documents(self) -> List[Dict[str, str]]:
        documents = []
        docs_path = self.config["data"]["documents_path"]

        for filename in os.listdir(docs_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(docs_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(
                        {
                            "id": filename.replace(".txt", ""),
                            "content": content,
                            "source": filename,
                        }
                    )

        return documents

    def embed_documents(self, documents: List[Dict[str, str]]) -> np.ndarray:
        texts = [doc["content"] for doc in documents]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, documents: List[Dict[str, str]]):
        embeddings_path = self.config["data"]["embeddings_path"]
        index_path = self.config["data"]["index_path"]

        np.save(embeddings_path, embeddings)

        index = [{"id": doc["id"], "source": doc["source"]} for doc in documents]
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def run(self):
        print("Loading documents...")
        documents = self.load_documents()
        print(f"Loaded {len(documents)} documents")

        print("Generating embeddings...")
        embeddings = self.embed_documents(documents)
        print(f"Generated embeddings with shape {embeddings.shape}")

        print("Saving embeddings...")
        self.save_embeddings(embeddings, documents)
        print("Embeddings saved successfully")

        return embeddings, documents
