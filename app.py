import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessing tools
model = joblib.load("model/model.pkl")
encoders_data = joblib.load("model/encoders.pkl")
label_encoders = encoders_data["encoders"]
scaler = encoders_data["scaler"]
target_encoder = encoders_data["target_encoder"]

st.set_page_config(page_title="💼 Salary Prediction App", page_icon="💼")
st.title("💼 Employee Salary Predection ")
st.write("Fill in the employee's details to predict if their salary exceeds $50K.")

def user_input():
    inputs = {}
    for col in model.feature_names_in_:
        if col in label_encoders:
            options = label_encoders[col].classes_
            inputs[col] = st.selectbox(f"{col}*", options)
        else:
            user_val = st.text_input(f"{col}*", placeholder="enter  value")
            inputs[col] = user_val
    return inputs

input_dict = user_input()

# Handle predict only if all required values are present
if st.button("Predict Salary"):
    try:
        # Convert input values
        processed_inputs = {}
        for col, val in input_dict.items():
            if col in label_encoders:
                processed_inputs[col] = val
            else:
                if val.strip() == "":
                    raise ValueError(f"Please enter a valid number for '{col}'")
                processed_inputs[col] = float(val)

        input_df = pd.DataFrame([processed_inputs])

        # Encode categorical
        for col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

        # Scale numeric
        numeric_cols = [col for col in input_df.columns if col not in label_encoders]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Reorder columns to match model
        input_df = input_df[model.feature_names_in_]

        # Predict
        prediction = model.predict(input_df)
        result = target_encoder.inverse_transform(prediction)[0]

        # Display result
        if result == ">50K":
            st.success("💰 Predicted Income: More than $50K")
        else:
            st.warning("🔻 Predicted Income: $50K or less")

        # Display input
        display_df = pd.DataFrame([input_dict])
        display_df["Predicted Income"] = result

        st.write("🔍 Prediction Details:")
        st.dataframe(display_df)

    except ValueError as e:
        st.error(str(e))
