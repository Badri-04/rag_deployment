import mlflow
import os
import yaml
from src.embedder import DocumentEmbedder
from src.retriever import DocumentRetriever
from src.generator import ResponseGenerator
from typing import Dict

class RAGPipeline:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.embedder = DocumentEmbedder(config_path)
        self.retriever = DocumentRetriever(config_path)
        self.generator = ResponseGenerator(config_path)

        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', self.config['mlflow']['tracking_uri'])
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def initialize(self):
        print('Initializing RAG pipeline...')
        self.embedder.run()
        self.retriever.load_embeddings()
        self.retriever.load_documents()
        print('Pipeline initialized successfully')

    def query(self, question: str) -> Dict[str, any]:
        with mlflow.start_run(run_name='rag_query'):
            mlflow.log_param('question', question)
            mlflow.log_param('top_k', self.config['retriever']['top_k'])
            mlflow.log_param('similarity_threshold', self.config['retriever']['similarity_threshold'])

            retrieved_docs = self.retriever.retrieve(question)
            mlflow.log_metric('num_retrieved_docs', len(retrieved_docs))

            if retrieved_docs:
                avg_similarity = sum(doc['similarity'] for doc in retrieved_docs) / len(retrieved_docs)
                mlflow.log_metric('avg_similarity', avg_similarity)

            response = self.generator.generate_response(question, retrieved_docs)
            mlflow.log_param('response_length', len(response))

            return {
                'question': question,
                'answer': response,
                'retrieved_docs': retrieved_docs,
                'num_docs': len(retrieved_docs)
            }
