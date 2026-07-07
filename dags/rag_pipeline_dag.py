from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/opt/airflow')

from src.embedder import DocumentEmbedder
from src.pipeline import RAGPipeline
import mlflow

default_args = {
    'owner': 'badri',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'rag_pipeline_daily',
    default_args=default_args,
    description='Daily RAG pipeline for document embedding and model evaluation',
    schedule_interval='0 3 * * *',
    catchup=False
)

def check_for_new_documents(**context):
    import os
    docs_path = '/opt/airflow/data/documents'
    document_count = len([f for f in os.listdir(docs_path) if f.endswith('.txt')])

    context['ti'].xcom_push(key='document_count', value=document_count)
    print(f'Found {document_count} documents')

    return document_count > 0

def generate_embeddings(**context):
    embedder = DocumentEmbedder('/opt/airflow/config/config.yaml')
    embeddings, documents = embedder.run()

    context['ti'].xcom_push(key='num_embeddings', value=len(embeddings))
    print(f'Generated {len(embeddings)} embeddings')

def evaluate_retrieval_quality(**context):
    from src.retriever import DocumentRetriever

    retriever = DocumentRetriever('/opt/airflow/config/config.yaml')
    retriever.load_embeddings()
    retriever.load_documents()

    test_queries = [
        "How long does shipping take?",
        "What is the return policy?",
        "How do I reset my password?"
    ]

    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.set_experiment('rag_pipeline_evaluation')

    with mlflow.start_run(run_name='daily_evaluation'):
        total_docs_retrieved = 0
        avg_similarity_scores = []

        for query in test_queries:
            results = retriever.retrieve(query)
            total_docs_retrieved += len(results)

            if results:
                avg_sim = sum(r['similarity'] for r in results) / len(results)
                avg_similarity_scores.append(avg_sim)

        mlflow.log_metric('avg_docs_per_query', total_docs_retrieved / len(test_queries))
        mlflow.log_metric('avg_similarity', sum(avg_similarity_scores) / len(avg_similarity_scores))
        mlflow.log_metric('num_queries_tested', len(test_queries))

        print(f'Evaluation complete. Avg docs per query: {total_docs_retrieved / len(test_queries):.2f}')

def cleanup_old_embeddings(**context):
    import os
    import time

    embeddings_path = '/opt/airflow/data/embeddings'
    current_time = time.time()
    files_removed = 0

    for filename in os.listdir(embeddings_path):
        filepath = os.path.join(embeddings_path, filename)
        if filename.startswith('backup_'):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > 30 * 24 * 3600:
                os.remove(filepath)
                files_removed += 1

    print(f'Removed {files_removed} old backup files')

check_documents = PythonOperator(
    task_id='check_for_new_documents',
    python_callable=check_for_new_documents,
    provide_context=True,
    dag=dag,
)

generate_embeddings_task = PythonOperator(
    task_id='generate_embeddings',
    python_callable=generate_embeddings,
    provide_context=True,
    dag=dag,
)

evaluate_quality = PythonOperator(
    task_id='evaluate_retrieval_quality',
    python_callable=evaluate_retrieval_quality,
    provide_context=True,
    dag=dag,
)

cleanup = PythonOperator(
    task_id='cleanup_old_embeddings',
    python_callable=cleanup_old_embeddings,
    provide_context=True,
    dag=dag,
)

check_documents >> generate_embeddings_task >> evaluate_quality >> cleanup
