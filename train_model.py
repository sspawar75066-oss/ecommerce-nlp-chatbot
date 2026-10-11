import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


df = pd.read_csv("intents.csv")

X = df["text"]
y = df["intent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model training completed")
print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


labels = sorted(y.unique())

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("Confusion Matrix")
print("Labels:", labels)
print(matrix)


joblib.dump(model, "chatbot_model.joblib")

print("\nModel saved as chatbot_model.joblib")


sample_messages = [
    "Where can I track my package?",
    "I want to send my product back",
    "My card payment failed",
    "When will my refund arrive?"
]

print("\nSample Predictions:")

for message in sample_messages:
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    confidence = max(probabilities)

    print("\nMessage:", message)
    print("Predicted intent:", prediction)
    print("Confidence:", round(confidence * 100, 2), "%")


print("\nImportant words for each intent:")

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()

for index, intent in enumerate(classifier.classes_):
    top_indices = classifier.coef_[index].argsort()[-8:][::-1]
    top_words = feature_names[top_indices]

    print(intent, ":", ", ".join(top_words))