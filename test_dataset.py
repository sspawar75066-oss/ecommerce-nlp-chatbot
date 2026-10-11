import pandas as pd

df = pd.read_csv("intents.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nIntent counts:")
print(df["intent"].value_counts())

print("\nFirst five rows:")
print(df.head())