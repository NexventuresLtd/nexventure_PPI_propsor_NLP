from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Path to your fine-tuned model folder
MODEL_NAME = "models/cls"

# Load tokenizer and model from fine-tuned local directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Setup classification pipeline for ease of use
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Assuming labels from fine-tuning are exactly the same as your training labels
# If different, adjust CATEGORY_MAP accordingly
CATEGORY_MAP = {
    "technology": "IT Equipment",
    "furniture": "Office Furniture",
    "energy": "Energy Equipment",
    "general": "General Supplies"
}

def classify_text(text: str):
    preds = classifier(text, truncation=True)
    raw_pred = preds[0]  # take top prediction
    label = raw_pred["label"].lower()
    confidence = raw_pred["score"]
    
    # Map label to your category map, fallback to label if unmapped
    category = CATEGORY_MAP.get(label, label)
    
    # Return detailed info to make it easier to debug/use downstream
    return {
        "category": category,
        "model_label": label,
        "confidence": round(confidence, 4)
    }

# Sample usage to test
if __name__ == "__main__":
    test_text = "We need to purchase 10 new laptops for the IT department."
    print(classify_text(test_text))
