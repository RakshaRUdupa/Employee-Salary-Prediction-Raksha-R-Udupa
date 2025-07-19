# Streamlit app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
data = pd.read_csv("Salary Data.csv")

# Load trained model and column structure
model = joblib.load('salary_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# -------------------- Streamlit App UI --------------------

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Predictor")
st.markdown("Predict the estimated **monthly salary** based on employee details using Machine Learning.")

# -------------------- Sidebar Input --------------------

st.sidebar.header("Enter Employee Details")

def user_input_features():
    age = st.sidebar.slider("📅 Age", 18, 65, 30)
    experience = st.sidebar.slider("📈 Years of Experience", 0, 40, 5)
    gender = st.sidebar.radio("🧑‍🤝‍🧑 Gender", ["Male", "Female"])
    education = st.sidebar.selectbox("🎓 Education Level", ["Bachelors", "Masters", "PhD"])
    position = st.sidebar.selectbox("💼 Job Role", ["Data Scientist", "Software Engineer", "Data Analyst"])

    input_dict = {
        'Age': age,
        'Years of Experience': experience,
    }

    # Manual one-hot encoding for categorical columns
    for col in model_columns:
        if col.startswith("Gender_"):
            input_dict[col] = 1 if col == f"Gender_{gender}" else 0
        elif col.startswith("Education Level_"):
            input_dict[col] = 1 if col == f"Education Level_{education}" else 0
        elif col.startswith("Position_"):
            input_dict[col] = 1 if col == f"Position_{position}" else 0
        elif col not in input_dict:
            input_dict[col] = 0  # Default for missing columns

    return pd.DataFrame([input_dict])

input_df = user_input_features()

# -------------------- Prediction --------------------

if st.button("🔍 Predict Salary"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Monthly Salary: ₹{prediction[0]:,.2f}")

# -------------------- Data Visualization --------------------

st.markdown("---")
st.subheader("📊 Experience vs Salary in Dataset")

fig, ax = plt.subplots()
sns.barplot(x='Years of Experience', y='Salary', data=data, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

