import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load the transaction data
df = pd.read_csv("transactions.csv")

# Encode categorical variables
label_encoders = {}
for col in ["Device", "Location"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and labels
X = df.drop(["Transaction_ID", "Fraudulent"], axis=1)
y = df["Fraudulent"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and preprocessing tools
model_data = {
    "model": model,
    "scaler": scaler,
    "encoders": label_encoders
}

with open("trained_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("✅ Model trained and saved as trained_model.pkl")
