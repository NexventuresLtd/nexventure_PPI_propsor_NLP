from app.ner_model import extract_entities
from app.classifier_model import classify_text
def parse_procurement(text: str):
    entities = extract_entities(text)
    result = {
        "quantity": None,
        "brand": None,
        "model": None,
        "type": None,
        "category": classify_text(text)
    }

    for ent in entities:
        label = ent["entity"].upper()
        word = ent["word"]

        if "QUANTITY" in label:
            result["quantity"] = word
        elif "BRAND" in label:
            result["brand"] = word
        elif "MODEL" in label:
            if result["model"]:
                result["model"] += f" {word}"
            else:
                result["model"] = word
        elif "TYPE" in label:
            result["type"] = word

    return result
