"""
Script to extract text from PDFs.
- Reads all PDF files in the 'pdfs/' directory.
- Outputs text files to the 'data/' directory.
"""
import os
from PyPDF2 import PdfReader

PDF_DIR = "pdfs"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

for pdf_file in os.listdir(PDF_DIR):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        text_path = os.path.join(DATA_DIR, pdf_file.replace('.pdf', '.txt'))
        with open(text_path, 'w') as out_file:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                out_file.write(page.extract_text())
