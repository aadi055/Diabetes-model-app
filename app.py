import streamlit as st
import pandas as pd
import pickle

# --- Load the Saved Model ---
# We load the trained RandomForest model pipeline that was saved in the previous step.
try:
    with open('diabetes_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except FileNotFoundError:
    st.error("Error: 'diabetes_model.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()  # Stop the app if the model isn't found
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --- Web App Interface ---

# Set the title of the web app
st.title('AI-Powered Diabetes Risk Predictor')
st.write("This app predicts the likelihood of a person having diabetes based on their medical information. Please enter the details below.")

# Add an image for a better user experience
st.image("https://images.unsplash.com/photo-1579165466949-3180a3d056d5?q=80&w=2070",
         caption="Stay Healthy, Stay Informed")

st.header("Patient's Medical Information")

# --- User Input Fields ---
# Create input fields for all the features the model was trained on.
# We'll use columns to organize the layout neatly.
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        'Pregnancies', min_value=0, max_value=20, value=1, step=1, help="Number of times pregnant")
    glucose = st.slider('Glucose', min_value=0, max_value=200, value=120,
                        help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    blood_pressure = st.slider('Blood Pressure (mm Hg)', min_value=0,
                               max_value=130, value=70, help="Diastolic blood pressure")
    skin_thickness = st.slider('Skin Thickness (mm)', min_value=0,
                               max_value=100, value=20, help="Triceps skin fold thickness")

with col2:
    insulin = st.slider('Insulin (mu U/ml)', min_value=0,
                        max_value=900, value=80, help="2-Hour serum insulin")
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=32.0,
                          format="%.1f", help="Body mass index (weight in kg/(height in m)^2)")
    dpf = st.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.47,
                    help="A function that scores likelihood of diabetes based on family history")
    age = st.number_input('Age', min_value=1, max_value=120, value=33, step=1)

# --- Prediction Logic ---

# Create a button that, when clicked, will trigger the prediction
if st.button('**Predict Diabetes Risk**', help="Click to see the prediction"):
    # 1. Collect user input into a DataFrame
    # The feature names must match *exactly* what the model was trained on.
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    st.write("---")
    st.subheader("Input Values:")
    st.write(input_data)

    # 2. Use the loaded model to make a prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)  # Get probabilities

    # 3. Display the prediction to the user
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f'**High Risk of Diabetes**', icon="⚠️")
        st.write(
            f"The model predicts a **{prediction_proba[0][1]*100:.2f}% probability** of having diabetes.")
        st.warning(
            "Please consult a healthcare professional for an accurate diagnosis and advice.")
    else:
        st.success(f'**Low Risk of Diabetes**', icon="✅")
        st.write(
            f"The model predicts a **{prediction_proba[0][0]*100:.2f}% probability** of not having diabetes.")
        st.info(
            "This is a good sign, but remember to maintain a healthy lifestyle and have regular check-ups.")

st.write("---")
st.markdown(
    "Developed for a portfolio project. **Disclaimer:** This is not a medical tool.")
