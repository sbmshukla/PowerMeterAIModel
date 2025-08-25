import streamlit as st
import pickle
import pandas as pd
import time

# Sidebar header
st.sidebar.header("Predict Energy Consumption")

# Model selection
model_choice = st.sidebar.selectbox("Choose Model", ["Linear Regression (99% Over-fit)", "ElasticNetCV (Adjusted For New Data)"])

# User input form
with st.sidebar.form(key="energy_form"):
    building_type = st.selectbox("Building Type", ["Residential", "Commercial", "Industrial"])
    square_footage = st.number_input("Square Footage", min_value=100, max_value=10000, value=500)
    num_occupants = st.number_input("Number of Occupants", min_value=1, max_value=500, value=10)
    appliances_used = st.number_input("Appliances Used", min_value=0, max_value=50, value=5)
    submitted = st.form_submit_button("Predict")

# Prediction with animation
if submitted:
    input_df = pd.DataFrame({
        'Building Type': [building_type],
        'Square Footage': [square_footage],
        'Number of Occupants': [num_occupants],
        'Appliances Used': [appliances_used]
    })

    # Spinner animation while loading the model and predicting
    with st.spinner('Predicting energy consumption...'):
        time.sleep(1)  # simulate processing time
        if model_choice == "Linear Regression":
            model = pickle.load(open('models/lr_model.pkl', 'rb'))
        else:
            model = pickle.load(open('models/enc_model.pkl', 'rb'))

        predicted_energy = model.predict(input_df)[0]

    # Animated metric display
    metric_placeholder = st.empty()
    for i in range(0, int(predicted_energy) + 1, max(1, int(predicted_energy / 50))):
        metric_placeholder.metric(label=f"Predicted Energy Consumption ({model_choice})", value=f"{i} kWh")
        time.sleep(0.02)
