import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

st.title("🫀 Heart Failure Prediction App")
st.markdown("Predict the risk of heart failure based on clinical parameters using Logistic Regression.")

# Prepare data
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"✅ Model trained with accuracy: **{accuracy:.2f}**")

# Sidebar Inputs
st.sidebar.header("📝 Enter Patient Clinical Data")

# Grouped and user-friendly inputs
age = st.sidebar.number_input("Age (years)", min_value=30, max_value=100, value=60, step=1)
sex = st.sidebar.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

st.sidebar.markdown("---")

anaemia = st.sidebar.radio("Anaemia (low red blood cells)?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
diabetes = st.sidebar.radio("Diabetes?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
high_blood_pressure = st.sidebar.radio("High Blood Pressure?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
smoking = st.sidebar.radio("Smoking?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

st.sidebar.markdown("---")

creatinine_phosphokinase = st.sidebar.slider("Creatinine Phosphokinase (mcg/L)", 23, 8000, 250)
ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", 10, 80, 38)
platelets = st.sidebar.slider("Platelets (kiloplatelets/mL)", 25000, 850000, 260000)
serum_creatinine = st.sidebar.slider("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.0)
serum_sodium = st.sidebar.slider("Serum Sodium (mEq/L)", 110, 150, 137)
time = st.sidebar.slider("Follow-up Period (days)", 0, 300, 120)

# Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "anaemia": [anaemia],
    "creatinine_phosphokinase": [creatinine_phosphokinase],
    "diabetes": [diabetes],
    "ejection_fraction": [ejection_fraction],
    "high_blood_pressure": [high_blood_pressure],
    "platelets": [platelets],
    "serum_creatinine": [serum_creatinine],
    "serum_sodium": [serum_sodium],
    "sex": [sex],
    "smoking": [smoking],
    "time": [time]
})

# Standardize input
input_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Display result
st.markdown("### 🧪 Prediction Result:")
if prediction == 1:
    st.error(f"⚠️ High Risk of Heart Failure! (Probability: {probability*100:.2f}%)")
else:
    st.success(f"✅ Low Risk of Heart Failure. (Probability: {probability*100:.2f}%)")

