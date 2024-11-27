"""
Script to fine-tune a transformer model on preprocessed text chunks.
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler

# Constants
PREPROCESSED_DIR = "data/preprocessed"
MODEL_NAME = "distilbert-base-uncased"  # Replace with your preferred model
OUTPUT_DIR = "models"
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LENGTH = 512

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text_dir):
        self.text_files = [os.path.join(text_dir, f) for f in os.listdir(text_dir) if f.endswith(".txt")]

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, idx):
        with open(self.text_files[idx], "r") as f:
            text = f.read()
        return text

# Prepare data
def preprocess_batch(batch_texts):
    """
    Tokenizes and processes a batch of texts into input tensors.
    """
    return tokenizer(
        batch_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# Train function
def train_model():
    # Load dataset and DataLoader
    dataset = TextDataset(PREPROCESSED_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # Adjust `num_labels` if needed
	#model.to('cpu')
    model.train()  # Set model to training mode

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * EPOCHS,
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0
        for batch in dataloader:
            batch_texts = list(batch)
            inputs = preprocess_batch(batch_texts)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            # Labels (dummy for now, replace with meaningful labels if available)
            labels = torch.zeros(len(batch_texts), dtype=torch.long).to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} loss: {epoch_loss / len(dataloader)}")

    # Save the fine-tuned model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()

