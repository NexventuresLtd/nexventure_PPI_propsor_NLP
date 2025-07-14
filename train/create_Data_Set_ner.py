import json
import random
from tqdm import tqdm
import re

quantities = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]

brands_tech = ["HP", "Dell", "Lenovo", "Samsung", "Apple", "Canon", "Acer"]
models_tech = ["LaserJet Pro M404dn", "Latitude 3420", "MacBook Pro", "Galaxy Tab S7", "ThinkPad X1", "Inspiron 15"]
items_tech = ["printers", "laptops", "tablets", "monitors", "desktop computers", "network routers", "projectors"]

brands_furniture = ["IKEA", "Steelcase", "Herman Miller", "Haworth", "LocalCraft", "Rwanda Furnishings"]
items_furniture = ["office chairs", "desks", "filing cabinets", "conference tables", "cubicle dividers", "bookshelves", "reception counters"]

brands_energy = ["SolarTech", "Rwanda Energy Co.", "GreenPower", "EcoSolar"]
items_energy = ["solar panels", "LED bulbs", "battery kits", "ceiling fans", "street lights", "inverters", "solar water heaters"]

brands_stationery = ["Pilot", "Bic", "Staples", "Rwanda Supplies"]
items_stationery = ["pens", "notebooks", "file folders", "whiteboards", "markers", "staplers", "paper reams"]

categories = {
    "technology": (brands_tech, models_tech, items_tech),
    "furniture": (brands_furniture, [], items_furniture),
    "energy": (brands_energy, [], items_energy),
    "stationery": (brands_stationery, [], items_stationery)
}

templates = [
    "Supply and delivery of {qty} {brand} {model} {item}",
    "Purchase of {qty} {brand} {model} {item} for government office",
    "Procurement of {qty} {brand} {item} to equip new offices",
    "Request for quotation: {qty} units of {brand} {model} {item}",
    "Acquisition of {qty} {item} ({brand} {model}) for national projects",
    "Supply of {qty} {item} by {brand} for upcoming events",
    "Provision of {qty} {brand} {model} {item} to regional offices",
    "Purchase order for {qty} {item} ({brand}) for public schools",
    "Urgent supply of {qty} {brand} {model} {item} to health centers",
    "Request for tender: {qty} {item} ({brand} {model}) for city council"
]

def tokenize(text):
    # Simple whitespace + punctuation tokenizer; you can replace with more advanced tokenizer if needed
    tokens = re.findall(r"\w+|\S", text)
    return tokens

def label_tokens(tokens, qty, brand, model, item):
    labels = ["O"] * len(tokens)

    def find_sublist(sublist):
        # Finds start index of sublist in tokens or returns -1
        for i in range(len(tokens) - len(sublist) + 1):
            if tokens[i:i+len(sublist)] == sublist:
                return i
        return -1

    # Label quantity (always numeric)
    qty_str = str(qty)
    qty_tokens = [qty_str]
    idx = find_sublist(qty_tokens)
    if idx != -1:
        labels[idx] = "B-QUANTITY"

    # Label brand (can be multiple tokens)
    if brand:
        brand_tokens = brand.split()
        idx = find_sublist(brand_tokens)
        if idx != -1:
            labels[idx] = "B-BRAND"
            for j in range(1, len(brand_tokens)):
                if idx + j < len(labels):
                    labels[idx + j] = "I-BRAND"

    # Label model (can be multiple tokens)
    if model:
        model_tokens = model.split()
        idx = find_sublist(model_tokens)
        if idx != -1:
            labels[idx] = "B-MODEL"
            for j in range(1, len(model_tokens)):
                if idx + j < len(labels):
                    labels[idx + j] = "I-MODEL"

    # Label item/type (can be multiple tokens)
    if item:
        item_tokens = item.split()
        idx = find_sublist(item_tokens)
        if idx != -1:
            labels[idx] = "B-TYPE"
            for j in range(1, len(item_tokens)):
                if idx + j < len(labels):
                    labels[idx + j] = "I-TYPE"

    return labels

dataset = []
num_samples = 10000  # Change to 1000000 if you want, but 10k is a good start for testing

for _ in tqdm(range(num_samples)):
    category = random.choice(list(categories.keys()))
    qty = random.choice(quantities)
    brands, models, items = categories[category]
    brand = random.choice(brands) if brands else ""
    model = random.choice(models) if models else ""
    item = random.choice(items)
    template = random.choice(templates)

    # Build sentence using template, carefully omitting model if empty
    if model:
        text = template.format(qty=qty, brand=brand, model=model, item=item)
    else:
        text = template.replace("{model}", "").format(qty=qty, brand=brand, item=item).replace("  ", " ").strip()

    tokens = tokenize(text)
    labels = label_tokens(tokens, qty, brand, model, item)

    dataset.append({
        "tokens": tokens,
        "labels": labels
    })

# Save to JSON
with open("train/ner_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} samples for NER dataset.")
