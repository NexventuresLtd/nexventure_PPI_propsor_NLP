from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import numpy as np

# Model to fine-tune
MODEL_NAME = "xlm-roberta-base"

# Load custom dataset (adjust path as needed)
raw_datasets = load_dataset("json", data_files="train/ner_dataset.json")

# BIO labels used in your dataset
label_list = [
    "O", 
    "B-QUANTITY", "I-QUANTITY",
    "B-BRAND", "I-BRAND",
    "B-MODEL", "I-MODEL",
    "B-TYPE", "I-TYPE"
]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize and align labels for each token
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",  # Ensures batching works
        max_length=128
    )
    
    labels = []
    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id.get(label_seq[word_idx], 0))
            else:
                # Use I- prefix for subsequent word pieces
                curr_label = label_seq[word_idx]
                if curr_label.startswith("B-"):
                    i_label = "I-" + curr_label[2:]
                else:
                    i_label = curr_label
                label_ids.append(label_to_id.get(i_label, 0))
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply the function to your dataset
tokenized_dataset = raw_datasets.map(tokenize_and_align_labels, batched=True)

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id_to_label,
    label2id=label_to_id
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./ner_output",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

# Padding collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Setup trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save model
model.save_pretrained("./models/ner")
tokenizer.save_pretrained("./models/ner")

print("✅ NER model trained and saved to ./models/ner")
