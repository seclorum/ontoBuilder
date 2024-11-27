# Makefile for automating the AI PDF project

.PHONY: all extract_text preprocess index query train

# Activate the virtual environment
VENV = source venv/bin/activate && 

all: extract_text preprocess index query train

prepare:
	./cp_random.sh

extract_text:
	$(VENV) python scripts/extract_text.py

preprocess:
	$(VENV) python scripts/preprocess.py

index:
	$(VENV) python scripts/create_index.py

query:
	$(VENV) python scripts/query_model.py

train:
	$(VENV) python scripts/train_model.py

clean:
	rm -rf {data,pdfs,models,embeddings}
	mkdir -p {data,pdfs,models,embeddings}
