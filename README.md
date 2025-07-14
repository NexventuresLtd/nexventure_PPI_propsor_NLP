# Umucyo NLP

A natural language processing (NLP) project designed to extract **structured information** from procurement texts in Rwanda, using **Named Entity Recognition (NER)** and **Text Classification**.

---
**Note**: this the MVP using sample dataset
## 📦 Repository

**GitHub Repo:**  
➡️ https://github.com/nexventures-ltd/nexventure_PPI_propsor_NLP

Clone it:

```bash
git clone https://github.com/nexventures-ltd/nexventure_PPI_propsor_NLP.git
cd nexventure_PPI_propsor_NLP
````

---

## 📁 Project Structure

```
nexventure_PPI_propsor_NLP/
├── app/                    # Core app logic
│   ├── ner_model.py        # NER inference logic
│   ├── classifier_model.py # Classification inference logic
│   ├── parser.py           # Combines NER + classifier into unified output
│   └── sample_data.py      # Example test texts
│
├── train/                  # Training & dataset creation
│   ├── fine_tune_ner.py          # Train NER model
│   ├── fine_tune_cls.py          # Train classifier model
│   ├── ner_dataset.json          # NER training data
│   ├── cls_dataset.json          # Classification training data
│   ├── create_data_set_ner.py    # Generates 10K+ NER training examples
│   └── create_data_set_cls.py    # Generates 10K+ classification examples
│
├── api/
│   └── main.py             # FastAPI server for inference
│
├── environment.yml         # Conda environment definition
└── README.md               # This file
```

---

## 🧪 Quick Start

### 1. 🐍 Create Conda Environment

```bash
conda env create -f environment.yml
conda activate umucyo_nlp
```

---

### 2. 📊 Generate Training Datasets

Run both scripts to generate **NER** and **classification** datasets:

```bash
python train/create_data_set_ner.py
python train/create_data_set_cls.py
```

This will generate `ner_dataset.json` and `cls_dataset.json` in the `train/` folder.

---

### 3. 🧠 Fine-Tune the Models

Train the **classification** model:

```bash
python train/fine_tune_cls.py
```

Train the **NER** model:

```bash
python train/fine_tune_ner.py
```

The fine-tuned models will be saved in:

* `models/cls/` for the classifier
* `models/ner/` for NER

---

### 4. 🚀 Run the FastAPI Server

```bash
uvicorn api.main:app --reload
```

You can now send predictions via:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text":"Supply and delivery of 10 HP LaserJet Pro M404dn printers"}'
```

---

### ✅ Sample Output

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

## 🧠 Model Summary

* **Model architecture**: `xlm-roberta-base`
* **Tasks**:

  * Classification: `technology`, `furniture`, `energy`, .
  * NER labels: `B-QUANTITY`, `B-BRAND`, `B-MODEL`, `B-TYPE`, .

---

## 🛠 Troubleshooting

* `null` values in output? → Make sure models are trained and saved under `models/`
* Datasets missing? → Re-run:

  ```bash
  python train/create_data_set_ner.py
  python train/create_data_set_cls.py
  ```

---

## 📬 Contributing

Pull requests and suggestions are welcome. Please fork and submit a PR or open an issue for discussion.

---

## 📜 License

MIT License – feel free to use and modify.

