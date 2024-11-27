#!/bin/bash

# Set up a Python project for training an AI model on local PDFs.
# This script sets up the required Python environment, including tools
#
# Note that the venv directory is .gitignore'd!
#

# Step 1: Create project directory
PROJECT_NAME="ontoBuilderML"
mkdir -p $PROJECT_NAME/{data,pdfs,scripts,models,embeddings}
cd $PROJECT_NAME

# Step 2: Create a Python virtual environment
echo "Setting up Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Step 3: Install required Python libraries
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install PyPDF2 pdfplumber pytesseract nltk numpy==1.26.4 scipy scikit-learn transformers faiss-cpu torch pycryptodome==3.15.0


# Final instructions
echo "Project setup complete. Run 'make' or specific targets (e.g., 'make extract_text') to execute."

