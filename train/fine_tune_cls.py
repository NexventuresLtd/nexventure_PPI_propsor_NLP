from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

MODEL = "xlm-roberta-base"
raw_datasets = load_dataset("json", data_files="train/cls_dataset.json")

# Create label mappings
label_list = sorted(list(set(example["label"] for example in raw_datasets["train"])))
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

print(f"Labels: {label_list}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_and_encode(batch):
    tokenized = tokenizer(batch["text"], padding=True, truncation=True)
    # Convert list of labels if any to int ids
    labels = []
    for label in batch["label"]:
        if isinstance(label, list):
            label = label[0]  # or handle multi-label case if you have one
        labels.append(label_to_id[label])
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = raw_datasets["train"].map(
    tokenize_and_encode,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=len(label_list),
    id2label=id_to_label,
    label2id=label_to_id,
)

training_args = TrainingArguments(
    output_dir="./cls_output",
    learning_rate=1e-5,
    per_device_train_batch_size=10,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("models/cls")
tokenizer.save_pretrained("models/cls")

print("Classification model fine-tuned and saved successfully.")
