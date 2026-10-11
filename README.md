# Case Study Report: AI-Powered E-Commerce NLP Customer Support Chatbot

## 1. Executive Summary
This project demonstrates an interactive, AI-driven NLP system designed to classify customer support inquiries and parse essential entities in real time. Built for an e-commerce platform handling thousands of daily queries, the system categorizes user messages, extracts actionable entities (Order IDs, Payment IDs, Amounts), highlights key predictive features, and provides a dynamic analytical UI.

---

## 2. Intent Categories & Case Study Target Queries
The model targets four core e-commerce customer support categories:

1. **Order Tracking (`order_tracking`):** *"Where is my order ORD12345?"*
2. **Product Returns (`return_request`):** *"I want to return my product"*
3. **Failed Payments (`payment_failed`):** *"My payment was unsuccessful"*
4. **Refund Status (`refund_status`):** *"When will I receive my refund back?"*

---

## 3. NLP Architecture & Technical Design

### A. Preprocessing & Tokenization
- Input text is cleaned, lowercased, and tokenized using word-boundary regular expressions (`\b[a-zA-Z0-9]+\b`).
- Character and token counts are computed instantly for dashboard insights.

### B. Intent Classification Pipeline
- **Vectorization:** TF-IDF Vectorizer with unigrams and bigrams (`ngram_range=(1,2)`) converts text into weighted numerical vectors while removing standard English stop words.
- **Classifier:** Logistic Regression trained on labeled customer queries predicts the target intent and outputs confidence probabilities (`predict_proba`).

### C. Named Entity Recognition (NER)
Custom regex extraction rules parse structural business entities from raw unstructured messages:
- **Order IDs:** Pattern matching `ORD-XXXXX` / `ORDER-XXXXX`
- **Payment IDs:** Pattern matching `PAY-XXXXX` / `TXN-XXXXX`
- **Monetary Amounts:** Pattern matching currency symbols (`$`, `₹`, `INR`, `Rs.`) followed by numeric values.

### D. Explainable Feature Analysis
The system identifies key influential keywords by inspecting class coefficients in the classifier model, filtering for exact feature matches present in the user's input.

---

## 4. System Implementation & Tech Stack

| Component | Technology / Framework | Role |
| :--- | :--- | :--- |
| **Backend API** | FastAPI + Uvicorn | RESTful endpoints (`/chat`) for inference |
| **NLP & ML Models** | Scikit-Learn + Joblib | TF-IDF Vectorizer & Logistic Regression pipeline |
| **Data Processing** | Pandas + Regex | Dataset management and regex-based NER |
| **Frontend UI** | HTML5, CSS3, JavaScript (Fetch API) | Two-column responsive dashboard interface |

---

## 5. Verification & Testing Results
All four target queries were verified against the live endpoint:
- **`order_tracking`**: Correctly identifies status requests and extracts order references (`ORD12345`).
- **`return_request`**: Flags return policies and triggers return window responses.
- **`payment_failed`**: Triggers payment troubleshooting guides.
- **`refund_status`**: Provides typical banking timeline estimates.

---

## 6. Project Structure
```text
ecommerce-chatbot/
├── intents.csv            # Training corpus (labeled dataset)
├── train_model.py         # Pipeline training script & joblib exporter
├── chatbot_model.joblib   # Trained ML model pipeline
├── main.py                # FastAPI server implementation
├── index.html             # Interactive frontend UI
└── requirements.txt       # Python environment dependencies