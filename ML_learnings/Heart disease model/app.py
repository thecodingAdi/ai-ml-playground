import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# ----------------------------
# Load and Apply Custom CSS with Background Image
# ----------------------------
# Load CSS file
with open("styles.css") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# Apply background image from URL
background_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQSAZy_NPB6pLZP-_hcpELj0kFVR0K2aANyZrtf6x7oWQ&s=10');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    background-repeat: no-repeat;
    position: relative;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 0;
    pointer-events: none;
}

[data-testid="stMainBlockContainer"] {
    background: transparent;
    position: relative;
    z-index: 1;
}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# ----------------------------
# Load model resources
# ----------------------------
try:
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_columns = joblib.load('columns.pkl')
except Exception as e:
    st.error(f"Failed to load model resources: {e}")
    st.stop()

st.title("Heart Disease Prediction App")
st.markdown("This app predicts the likelihood of heart disease based on user input.")

# ----------------------------
# User Inputs
# ----------------------------
age = st.slider("Age", 18, 100, 50)
sex = st.selectbox("Sex", options=["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", options=[
    "Typical Angina(TA)",
    "Atypical Angina(ATA)",
    "Non-anginal Pain(NAP)",
    "Asymptomatic(ASY)"
])

resting_blood_pressure = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
resting_ecg = st.selectbox("Resting ECG Results", options=[
    "Normal",
    "ST-T Wave Abnormality",
    "Left Ventricular Hypertrophy"
])

max_heart_rate = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
oldpeak = st.slider("Oldpeak (ST depression induced by exercise)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[
    "Upsloping",
    "Flat",
    "Downsloping"
])

# ----------------------------
# Mapping UI labels -> dataset's category codes
# ----------------------------
chest_pain_map = {
    "Typical Angina(TA)": "TA",
    "Atypical Angina(ATA)": "ATA",
    "Non-anginal Pain(NAP)": "NAP",
    "Asymptomatic(ASY)": "ASY"
}

resting_ecg_map = {
    "Normal": "Normal",
    "ST-T Wave Abnormality": "ST",
    "Left Ventricular Hypertrophy": "LVH"
}

st_slope_map = {
    "Upsloping": "Up",
    "Flat": "Flat",
    "Downsloping": "Down"
}

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict"):

    # Step 1: Raw input matching original (pre-dummy) dataset columns/names
    raw_input = pd.DataFrame({
        'Age': [age],
        'Sex': ["M" if sex == "Male" else "F"],
        'ChestPainType': [chest_pain_map[chest_pain_type]],
        'RestingBP': [resting_blood_pressure],
        'Cholesterol': [cholesterol],
        'FastingBS': [1 if fasting_blood_sugar == "Yes" else 0],
        'RestingECG': [resting_ecg_map[resting_ecg]],
        'MaxHR': [max_heart_rate],
        'ExerciseAngina': ["Y" if exercise_induced_angina == "Yes" else "N"],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_map[st_slope]]
    })

    # Step 2: One-hot encode exactly like training (drop_first=True)
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    input_df = pd.get_dummies(raw_input, columns=categorical_cols, drop_first=True)

    # Step 3: Add any missing expected columns as 0
    #    (e.g., if user picked the baseline category, its dummy column won't exist)
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Step 4: Drop extras and reorder to match training exactly
    input_df = input_df[expected_columns]

    # Step 5: Scale + Predict
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        if prediction == 1:
            st.error("⚠️ High risk of heart disease. Please consult a healthcare professional.")
        else:
            st.success("✅ Low risk of heart disease. Keep up the healthy lifestyle!")

        # Optional debug view — remove before deploying
        with st.expander("Debug: Model Input"):
            st.write(input_df)