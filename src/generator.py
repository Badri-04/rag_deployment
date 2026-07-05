import yaml
from typing import List, Dict
import os
from openai import OpenAI


class ResponseGenerator:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = self.config["generator"]["model_name"]
        self.max_tokens = self.config["generator"]["max_tokens"]
        self.temperature = self.config["generator"]["temperature"]

    def generate_response(
        self, query: str, retrieved_docs: List[Dict[str, any]]
    ) -> str:
        if not retrieved_docs:
            return "I could not find relevant information to answer your question."

        context = "\n\n".join(
            [
                f"Document {i + 1} (similarity: {doc['similarity']:.2f}):\n{doc['content']}"
                for i, doc in enumerate(retrieved_docs)
            ]
        )

        prompt = f"""You are a helpful customer support assistant.
        Use the following documentation to answer the customer's question.
        If the documentation does not contain enough information, say so clearly.

        Documentation:
        {context}

        Customer Question: {query}

        Answer:"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful customer support assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message.content
