"""
Script to preprocess extracted text.
- Tokenizes, removes stopwords, and cleans text data.
- Outputs preprocessed chunks to 'data/'.
"""
import os
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

DATA_DIR = "data"
PREPROCESSED_DIR = "data/preprocessed"

os.makedirs(PREPROCESSED_DIR, exist_ok=True)

for text_file in os.listdir(DATA_DIR):
    if text_file.endswith(".txt"):
        input_path = os.path.join(DATA_DIR, text_file)
        output_path = os.path.join(PREPROCESSED_DIR, text_file)
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            text = infile.read()
            tokens = word_tokenize(text)
            processed_text = " ".join(tokens)
            outfile.write(processed_text)
