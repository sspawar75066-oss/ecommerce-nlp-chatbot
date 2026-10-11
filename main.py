from pathlib import Path
import re

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "chatbot_model.joblib"
DATASET_PATH = BASE_DIR / "intents.csv"


model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="E-commerce NLP Customer Support Chatbot",
    description="Intent classification and NLP analysis for customer support messages.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    message: str


def tokenize_text(text: str):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def extract_entities(text: str):
    order_ids = re.findall(
        r"\b(?:ORD|ORDER)[-_]?\d{3,}\b",
        text.upper()
    )

    payment_ids = re.findall(
        r"\b(?:PAY|PAYMENT|TXN|TRANSACTION)[-_]?\d{3,}\b",
        text.upper()
    )

    amounts = re.findall(
        r"(?:₹|Rs\.?|INR|\$)\s?\d+(?:\.\d{1,2})?",
        text,
        flags=re.IGNORECASE
    )

    return {
        "order_ids": order_ids,
        "payment_ids": payment_ids,
        "amounts": amounts
    }


def get_important_keywords(text: str, intent: str):
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    tokens = set(tokenize_text(text))
    feature_names = vectorizer.get_feature_names_out()

    class_index = list(classifier.classes_).index(intent)
    top_indices = classifier.coef_[class_index].argsort()[-20:][::-1]

    important_words = []

    for index in top_indices:
        feature = feature_names[index]
        feature_tokens = set(feature.split())

        if feature_tokens.intersection(tokens):
            important_words.append(feature)

    return important_words[:8]


def find_response(intent: str):
    try:
        import pandas as pd

        data = pd.read_csv(DATASET_PATH)
        matching_rows = data[data["intent"] == intent]

        if not matching_rows.empty:
            return matching_rows.iloc[0]["response"]

    except Exception:
        pass

    return "Please contact customer support for more assistance."


@app.get("/")
def home():
    return {
        "message": "E-commerce NLP Chatbot API is running",
        "docs": "/docs",
        "endpoint": "/chat"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    confidence = max(probabilities)

    tokens = tokenize_text(message)
    entities = extract_entities(message)
    important_keywords = get_important_keywords(
        message,
        prediction
    )

    response = find_response(prediction)

    return {
        "message": message,
        "predicted_intent": prediction,
        "confidence": round(float(confidence), 4),
        "confidence_percentage": round(float(confidence) * 100, 2),
        "tokens": tokens,
        "token_count": len(tokens),
        "character_count": len(message),
        "important_keywords": important_keywords,
        "entities": entities,
        "bot_response": response
    }