from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

LABELS = [
    "O",
    "B-QUANTITY", "I-QUANTITY",
    "B-BRAND", "I-BRAND",
    "B-MODEL", "I-MODEL",
    "B-TYPE", "I-TYPE"
]

label_to_id = {label: i for i, label in enumerate(LABELS)}
id_to_label = {i: label for label, i in label_to_id.items()}

MODEL_PATH = "models/ner"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(LABELS),
    id2label=id_to_label,
    label2id=label_to_id
)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities(text: str):
    entities = ner_pipeline(text)
    print("Raw NER output:", entities)
    return [
        {
            "entity": ent["entity_group"],
            "word": ent["word"],
            "score": round(ent["score"], 3)
        }
        for ent in entities
    ]
