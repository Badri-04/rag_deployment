import mlflow
import os
import yaml
from src.embedder import DocumentEmbedder
from src.retriever import DocumentRetriever
from src.generator import ResponseGenerator
from typing import Dict
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

query_counter = Counter('rag_queries_total', 'Total number of RAG queries')
query_duration = Histogram('rag_query_duration_seconds', 'Time spent processing queries')
retrieval_quality = Gauge('rag_retrieval_similarity_avg', 'Average similarity score of retrieved documents')
num_retrieved_docs = Histogram('rag_num_retrieved_docs', 'Number of documents retrieved per query')

class RAGPipeline:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.embedder = DocumentEmbedder(config_path)
        self.retriever = DocumentRetriever(config_path)
        self.generator = ResponseGenerator(config_path)

        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", self.config["mlflow"]["tracking_uri"]
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

    def initialize(self):
        print("Initializing RAG pipeline...")
        self.embedder.run()
        self.retriever.load_embeddings()
        self.retriever.load_documents()
        print("Pipeline initialized successfully")

    def query(self, question: str) -> Dict[str, any]:
        start_time = time.time()
        query_counter.inc()

        with mlflow.start_run(run_name='rag_query'):
            mlflow.log_param('question', question)

            retrieved_docs = self.retriever.retrieve(question)
            num_retrieved_docs.observe(len(retrieved_docs))
            mlflow.log_metric('num_retrieved_docs', len(retrieved_docs))

            if retrieved_docs:
                avg_similarity = sum(doc['similarity'] for doc in retrieved_docs) / len(retrieved_docs)
                retrieval_quality.set(avg_similarity)
                mlflow.log_metric('avg_similarity', avg_similarity)

            response = self.generator.generate_response(question, retrieved_docs)

            duration = time.time() - start_time
            query_duration.observe(duration)
            mlflow.log_metric('query_duration', duration)

            return {
                'question': question,
                'answer': response,
                'retrieved_docs': retrieved_docs,
                'num_docs': len(retrieved_docs),
                'duration': duration
            }
        
    def metrics(self):
        return generate_latest()
