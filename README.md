# Umucyo NLP

A natural language processing project designed for intelligent analysis of **procurement texts** using **Named Entity Recognition (NER)** and **Text Classification** models.

---

## 📦 Repository

**GitHub Repo**:  
`https://github.com/nexventures-ltd/nexventure_PPI_propsor_NLP`

To clone:

```bash
git clone https://github.com/nexventures-ltd/nexventure_PPI_propsor_NLP
cd nexventure_PPI_propsor_NLP
````

---

## 📁 Project Structure

```bash
.
├── app/                    # Core application logic
│   ├── ner_model.py        # NER inference pipeline
│   ├── classifier_model.py # Classification pipeline
│   ├── parser.py           # Combines NER + classifier for final output
│   └── sample_data.py      # Example texts to test pipeline
│
├── train/                  # Training scripts and datasets
│   ├── fine_tune_ner.py    # Script to fine-tune NER model
│   ├── fine_tune_cls.py    # Script to fine-tune classifier model
│   ├── ner_dataset.json    # Dataset for NER (tokens + labels)
│   ├── cls_dataset.json    # Dataset for classification (text + category)
│   ├── create_data_set_ner.py   # Generates NER dataset (if missing)
│   └── create_data_set_cls.py   # Generates classification dataset (if missing)
│
├── api/
│   └── main.py             # FastAPI app for live API predictions
│
├── requirements.txt        # Python requirements (if using pip)
└── README.md
```

---

## 🧪 Quick Start

### 1. 🐍 Create Conda Environment

```bash
conda create -n umucyo_nlp python=3.10
conda activate umucyo_nlp
```

Then install dependencies:

```bash
# via conda
conda install pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge transformers datasets fastapi uvicorn scikit-learn pandas tqdm

# via pip
pip install accelerate sentencepiece python-multipart
```

---

### 2. 📊 Generate Dataset (if missing)

If your datasets (`ner_dataset.json`, `cls_dataset.json`) are missing or broken:

```bash
python train/create_data_set_ner.py
python train/create_data_set_cls.py
```

---

### 3. 🧠 Train Models

You can fine-tune both NER and classifier models using:

```bash
# Train classification model
python train/fine_tune_cls.py

# Train NER model
python train/fine_tune_ner.py
```

This will save models into `./models/cls` and `./models/ner`.

---

### 4. 🚀 Run the FastAPI Server

```bash
uvicorn api.main:app --reload
```

Access the API at:
`http://127.0.0.1:8000`

Test via:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text":"Supply and delivery of 10 HP LaserJet Pro M404dn printers"}'
```

---

### 5. ✅ Example Output

```json
{
  "input": "Supply and delivery of 10 HP LaserJet Pro M404dn printers",
  "structured": {
    "quantity": "10",
    "brand": "HP",
    "model": "LaserJet Pro M404dn",
    "type": "printers",
    "category": {
      "category": "IT Equipment",
      "model_label": "technology",
      "confidence": 0.9983
    }
  }
}
```

---

## 🧠 Model Info

* Classification model: Fine-tuned `xlm-roberta-base` on procurement category texts (`technology`, `furniture`, `energy`, etc.)
* NER model: Fine-tuned `xlm-roberta-base` to extract:

  * `B-QUANTITY`, `B-BRAND`, `B-MODEL`, `B-TYPE`, etc.

---

## 🛠 Troubleshooting

* ❌ If your output shows `null` values → Re-check model paths or re-run training
* ❌ Missing tokens/labels in output → Ensure your dataset matches the format
* ✅ Use the `create_data_set_*.py` generators to bootstrap your training files

---

## 👥 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) file for details.

