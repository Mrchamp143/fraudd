import pandas as pd
import numpy as np
import pickle

# Load trained model and preprocessing tools
with open("trained_model.pkl", "rb") as file:
    data = pickle.load(file)

model = data["model"]
scaler = data["scaler"]
label_encoders = data["encoders"]

# Input transaction details
transaction_id = input("Enter Transaction ID: ")
amount = float(input("Enter Amount: "))
device = input("Enter Device: ")
location = input("Enter Location: ")

# Prepare the input dataframe
input_data = pd.DataFrame([{
    "Transaction_ID": transaction_id,
    "Amount": amount,
    "Device": device,
    "Location": location
}])

# Encode categorical variables
for col in ["Device", "Location"]:
    if col in input_data.columns:
        if input_data[col][0] in label_encoders[col].classes_:
            input_data[col] = label_encoders[col].transform([input_data[col][0]])
        else:
            print(f"⚠️ Unknown {col}: '{input_data[col][0]}'. Assigning default encoding (-1).")
            input_data[col] = -1

# Prepare features
X_input = input_data.drop(["Transaction_ID"], axis=1)

# Standardize features
X_scaled = scaler.transform(X_input)

# Predict fraud status
prediction = model.predict(X_scaled)[0]

# Show the result
print("\n🧾 Transaction Summary:")
print(input_data.assign(Fraudulent=bool(prediction)))

if prediction == 1:
    print("\n🚨 Alert: This transaction is predicted to be FRAUDULENT!")
else:
    print("\n✅ This transaction is predicted to be SAFE.")
