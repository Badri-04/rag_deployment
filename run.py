from src.pipeline import RAGPipeline
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    pipeline = RAGPipeline()

    pipeline.initialize()

    questions = [
        "How long does standard shipping take?",
        "What is your return policy?",
        "How do I reset my password?",
        "Do you ship internationally?",
        "How long do refunds take to process?"
    ]

    for question in questions:
        print(f'\nQuestion: {question}')
        result = pipeline.query(question)
        print(f'Answer: {result["answer"]}')
        print(f'Retrieved {result["num_docs"]} documents')
        print('-' * 80)

if __name__ == '__main__':
    main()
