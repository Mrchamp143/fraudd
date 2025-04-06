import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# === PATHS ===
MODEL_PATH = os.path.join("..", "model", "trained_model.pkl")
CSV_PATH = os.path.join("..", "data", "transactions.csv")
HISTORY_PATH = os.path.join("..", "data", "history.csv")

# === LOAD MODEL ===
try:
    with open(MODEL_PATH, "rb") as file:
        data = pickle.load(file)

    model = data["model"]
    scaler = data["scaler"]
    label_encoders = data["encoders"]

    # Load original dataset
    df_transactions = pd.read_csv(CSV_PATH)

    # Streamlit layout
    st.set_page_config(page_title="UPI Fraud Detection", layout="centered")
    st.title("🔐 UPI Fraud Detection System")

    tab1, tab2, tab3 = st.tabs(["🧪 Predict Transaction", "📊 View Prediction History", "📁 Upload Bank CSV"])

    # === TAB 1: SINGLE TRANSACTION CHECK ===
    with tab1:
        st.subheader("📝 Enter Transaction Details")

        with st.form("fraud_form"):
            transaction_id = st.text_input("Transaction ID")
            amount = st.number_input("Amount (₹)", min_value=1.0)
            device = st.selectbox("Device", label_encoders["Device"].classes_)
            location = st.selectbox("Location", label_encoders["Location"].classes_)
            submit = st.form_submit_button("Check Fraud")

        if submit:
            if not transaction_id.strip():
                st.warning("⚠️ Please enter a valid Transaction ID.")
            elif amount <= 0:
                st.warning("⚠️ Please enter a valid amount.")
            else:
                match = df_transactions[
                    (df_transactions["Transaction_ID"] == transaction_id) &
                    (df_transactions["Amount"] == amount) &
                    (df_transactions["Device"] == device) &
                    (df_transactions["Location"] == location)
                ]

                if match.empty:
                    st.error("❌ Wrong details. Please enter values from the dataset.")
                else:
                    st.info("✅ Matching transaction found:")
                    st.dataframe(match)

                    # Prepare input
                    input_data = pd.DataFrame([{
                        "Transaction_ID": transaction_id,
                        "Amount": amount,
                        "Device": device,
                        "Location": location
                    }])

                    for col in ["Device", "Location"]:
                        input_data[col] = label_encoders[col].transform([input_data[col][0]])

                    X_input = input_data.drop("Transaction_ID", axis=1)
                    X_scaled = scaler.transform(X_input)
                    prediction = model.predict(X_scaled)[0]
                    is_fraud = bool(prediction)

                    st.subheader("🔍 Prediction Result")
                    st.write(input_data.assign(Fraudulent=is_fraud))

                    if is_fraud:
                        st.error("🚨 This transaction is **FRAUDULENT**.")
                    else:
                        st.success("✅ This transaction is **SAFE**.")

                    # Save to history
                    new_entry = {
                        "Transaction_ID": transaction_id,
                        "Amount": amount,
                        "Device": device,
                        "Location": location,
                        "Fraudulent": is_fraud
                    }

                    if os.path.exists(HISTORY_PATH):
                        df_history = pd.read_csv(HISTORY_PATH)
                        df_history = pd.concat([df_history, pd.DataFrame([new_entry])], ignore_index=True)
                    else:
                        df_history = pd.DataFrame([new_entry])

                    df_history.to_csv(HISTORY_PATH, index=False)
                    st.success("📁 Saved to prediction history.")

    # === TAB 2: VIEW HISTORY ===
    with tab2:
        st.subheader("📈 Past Predictions")

        if os.path.exists(HISTORY_PATH):
            df_history = pd.read_csv(HISTORY_PATH)

            col1, col2 = st.columns(2)
            with col1:
                filter_fraud = st.selectbox("Fraudulent?", options=["All", "Yes", "No"])
            with col2:
                filter_device = st.selectbox("Device Type", options=["All"] + sorted(df_history["Device"].unique()))

            filtered_df = df_history.copy()
            if filter_fraud == "Yes":
                filtered_df = filtered_df[filtered_df["Fraudulent"] == True]
            elif filter_fraud == "No":
                filtered_df = filtered_df[filtered_df["Fraudulent"] == False]

            if filter_device != "All":
                filtered_df = filtered_df[filtered_df["Device"] == filter_device]

            st.dataframe(filtered_df)
        else:
            st.info("No prediction history available yet.")

    # === TAB 3: UPLOAD BANK CSV ===
    with tab3:
        st.subheader("📤 Upload Bank Transaction CSV")

        uploaded_file = st.file_uploader("Upload CSV file with transaction data", type=["csv"])

        if uploaded_file:
            try:
                new_data = pd.read_csv(uploaded_file)

                required_cols = ["Transaction_ID", "Amount", "Device", "Location"]
                if not all(col in new_data.columns for col in required_cols):
                    st.error(f"❌ Uploaded file must contain columns: {', '.join(required_cols)}")
                else:
                    original_data = new_data.copy()

                    for col in ["Device", "Location"]:
                        new_data[col] = new_data[col].apply(
                            lambda x: label_encoders[col].transform([x])[0]
                            if x in label_encoders[col].classes_ else -1
                        )

                    X = new_data[["Amount", "Device", "Location"]]
                    X_scaled = scaler.transform(X)
                    predictions = model.predict(X_scaled)

                    original_data["Fraudulent"] = predictions
                    num_fraud = sum(predictions)

                    st.success(f"✅ Found {num_fraud} fraudulent transactions out of {len(predictions)} total.")

                    if num_fraud > 0:
                        st.subheader("🚨 Fraudulent Transactions Detected")
                        st.dataframe(original_data[original_data["Fraudulent"] == 1])
                    else:
                        st.info("🎉 No fraudulent transactions detected.")

            except Exception as e:
                st.error(f"❌ Error reading file: {e}")

except FileNotFoundError as e:
    st.error(f"❌ Required file not found: {e}")
