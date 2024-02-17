import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Function to load the model with caching
@st.cache_data
def load_model():
    try:
        with open('svc_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")

model = load_model()

# Title and subtitle without emojis
st.title("Drug Prediction App")
st.markdown("Predict the most suitable drug for a patient based on input features")

# User input features with appropriate labels
age = st.slider("Patient's Age", 1, 100, 50)
sex = st.radio("Patient's Gender", ["Male", "Female"])
bp = st.selectbox("Blood Pressure Level", ["Low", "Normal", "High"])
cl = st.selectbox("Cholesterol Level", ["Low", "High"])
na_to_k = st.slider("Na_to_K Ratio", 1.0, 45.0, 20.0)

# Display user inputs in an organized format
st.subheader("Patient Information:")
st.write(f"- Age: {age} years")
st.write(f"- Gender: {sex}")
st.write(f"- Blood Pressure: {bp}")
st.write(f"- Cholesterol Level: {cl}")
st.write(f"- Na_to_K Ratio: {na_to_k}")

# Map categorical features to numerical values
sex = 0 if sex == "Female" else 1
bp_mapping = {"Low": 0, "Normal": 1, "High": 2}
cl_mapping = {"Low": 0, "High": 1}
bp, cl = bp_mapping[bp], cl_mapping[cl]

# Mapping for displaying the drug name
drug_mapping = {0: 'Drug A', 1: 'Drug B', 2: 'Drug C', 3: 'Drug X', 4: 'Drug Y'}

# Prediction button without emojis
if st.button("Predict Drug"):
    with st.spinner("Making Prediction..."):
        # Make prediction using the model
        drug_code = model.predict([[age, sex, bp, cl, na_to_k]])[0]
        
        # Get the decision function scores for each class
        decision_scores = model.decision_function([[age, sex, bp, cl, na_to_k]])[0]
        
        # Apply softmax to convert decision function scores to probabilities
        confidence_scores = np.exp(decision_scores) / np.sum(np.exp(decision_scores))
        
        # Display the predicted drug without emojis
        st.success(f"The most suitable drug for the patient is: {drug_mapping[drug_code]}")
        
        # Display the confidence score on the next line
        st.write(f"Confidence Score: {confidence_scores[drug_code]:.2%}")

        # Display a brief explanation about the predicted drug
        drug_explanation = {
            'Drug A': 'This drug is commonly used for...',
            'Drug B': 'This drug is known to be effective in...',
            'Drug C': 'Patients prescribed with this drug typically experience...',
            'Drug X': 'This drug is recommended for...',
            'Drug Y': 'Patients taking this drug usually report...'
        }
        st.markdown(f"**Drug Explanation:**\n{drug_explanation[drug_mapping[drug_code]]}")

        # Display a bar chart of the prediction probability distribution
        fig, ax = plt.subplots()
        ax.bar([drug_mapping[i] for i in range(len(confidence_scores))], confidence_scores, color='skyblue')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probability Distribution')
        ax.legend(['Drug Probabilities'], loc='upper left')
        st.pyplot(fig)
