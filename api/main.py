from fastapi import FastAPI
from pydantic import BaseModel
from app.parser import parse_procurement

app = FastAPI(title="Umucyo NLP API")

class ProcurementInput(BaseModel):
    text: str

@app.post("/predict")
def predict_procurement(data: ProcurementInput):
    structured_output = parse_procurement(data.text)
    return {
        "input": data.text,
        "structured": structured_output
    }


# Run with:
# uvicorn api.main:app --reload
