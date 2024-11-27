"""
Script to query the AI system using the indexed data.
- Accepts a query from the user.
- Retrieves relevant chunks and answers using a local model.
"""

import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Constants
EMBEDDINGS_DIR = "embeddings"
PREPROCESSED_DIR = "data/preprocessed"
MODEL_NAME = "distilbert-base-uncased"  # Replace with your preferred model

# Load FAISS index
index = faiss.read_index(os.path.join(EMBEDDINGS_DIR, "faiss_index"))

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.to('cpu')

# Helper function: Generate query embedding
def generate_query_embedding(query):
    """
    Generates an embedding vector for the input query.
    """
    inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the last hidden state as the query embedding
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return query_embedding

# Helper function: Retrieve relevant chunks
def retrieve_relevant_chunks(query_embedding, top_k=5):
    """
    Retrieves the top_k most relevant chunks from the FAISS index.
    """
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:  # indices[0] contains the top_k indices
        if idx != -1:  # -1 indicates no match
            chunk_file = os.listdir(PREPROCESSED_DIR)[idx]  # Map index to chunk file
            with open(os.path.join(PREPROCESSED_DIR, chunk_file), 'r') as f:
                results.append(f.read())
    return results

# Main function: Query the model
def main():
    print("Welcome to the AI PDF Query System!")
    print("Type your query below (or 'exit' to quit):")

    while True:
        query = input("\nQuery: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        print("AA")
        # Generate embedding for the query
        query_embedding = generate_query_embedding(query)

        print("AB")
        # Retrieve relevant chunks
        top_k_results = retrieve_relevant_chunks(query_embedding)

        print("AC")
        print("\nTop results:")
        for i, result in enumerate(top_k_results, 1):
            print(f"\nResult {i}:\n{result}")

if __name__ == "__main__":
    main()

