import json
import random
from tqdm import tqdm  # for progress bar, install with: pip install tqdm

# Define quantities typical for procurement
quantities = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]

# Technology category (brands, models, items relevant to Rwanda)
brands_tech = ["HP", "Dell", "Lenovo", "Samsung", "Apple", "Canon", "Acer"]
models_tech = ["LaserJet Pro M404dn", "Latitude 3420", "MacBook Pro", "Galaxy Tab S7", "ThinkPad X1", "Inspiron 15"]
items_tech = ["printers", "laptops", "tablets", "monitors", "desktop computers", "network routers", "projectors"]

# Furniture category
brands_furniture = ["IKEA", "Steelcase", "Herman Miller", "Haworth", "LocalCraft", "Rwanda Furnishings"]
items_furniture = ["office chairs", "desks", "filing cabinets", "conference tables", "cubicle dividers", "bookshelves", "reception counters"]

# Energy category (solar, electrical supplies common in Rwanda)
brands_energy = ["SolarTech", "Rwanda Energy Co.", "GreenPower", "EcoSolar"]
items_energy = ["solar panels", "LED bulbs", "battery kits", "ceiling fans", "street lights", "inverters", "solar water heaters"]

# Stationery and general supplies (added to diversify)
brands_stationery = ["Pilot", "Bic", "Staples", "Rwanda Supplies"]
items_stationery = ["pens", "notebooks", "file folders", "whiteboards", "markers", "staplers", "paper reams"]

categories = {
    "technology": (brands_tech, models_tech, items_tech),
    "furniture": (brands_furniture, [], items_furniture),
    "energy": (brands_energy, [], items_energy),
    "stationery": (brands_stationery, [], items_stationery)
}

# Different phrase templates reflecting procurement language used in Rwanda
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

dataset = []

def choose_brand_model_item(category):
    brands, models, items = categories[category]
    brand = random.choice(brands) if brands else ""
    model = random.choice(models) if models else ""
    item = random.choice(items)
    return brand, model, item

num_samples = 10000

print("Generating dataset...")

for _ in tqdm(range(num_samples)):
    category = random.choice(list(categories.keys()))
    qty = random.choice(quantities)
    brand, model, item = choose_brand_model_item(category)
    template = random.choice(templates)

    # Format text with or without model, as some categories don't have models
    if model:
        text = template.format(qty=qty, brand=brand, model=model, item=item)
    else:
        # Remove model placeholder for categories without models (like furniture, stationery)
        text = template.replace("{model}", "").format(qty=qty, brand=brand, item=item).replace("  ", " ").strip()
    
    dataset.append({
        "text": text,
        "label": category
    })

# Save to file
output_path = "train/cls_dataset.json"
print(f"Saving dataset to {output_path} ...")
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} samples for classification dataset.")
