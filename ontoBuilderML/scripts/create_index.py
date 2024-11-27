"""
Script to create vector embeddings and an index for the text data.
- Uses FAISS for vector indexing.
- Outputs the index to 'embeddings/'.
"""
import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

DATA_DIR = "data/preprocessed"
EMBEDDINGS_DIR = "embeddings"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Load a local transformer model (e.g., "distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.to('cpu')


vectors = []
file_index = []

for text_file in os.listdir(DATA_DIR):
    with open(os.path.join(DATA_DIR, text_file), 'r') as infile:
        print("infile:", infile)
        text = infile.read()
        print("B")
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        print("C")
        outputs = model(**inputs)
        print("D")
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        print("E")
        vectors.append(embedding)
        print("F")
        file_index.append(text_file)
        print("G")

# Create FAISS index
print("AA")
index = faiss.IndexFlatL2(vectors[0].shape[1])
print("AB")
index.add(np.vstack(vectors))
print("AC")
faiss.write_index(index, os.path.join(EMBEDDINGS_DIR, "faiss_index"))
print("AD")
