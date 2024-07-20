import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'D:\\Bootcamp Project\\lung_cancer_analysis\\ML_Model\\knn_model.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")
    expected_columns = []  # Set a default to avoid breaking

def main():
    # Set the title of the web app
    st.title('Lung Cancer Risk Prediction')

    # Add a description
    st.write('Enter patient information to predict lung cancer risk level.')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Patient Information')

        # Add input fields for features
        age = st.slider('Age', 0, 100, 50)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        air_pollution = st.slider('Air Pollution Level', 1, 10, 5)
        alcohol_use = st.slider('Alcohol Use Level', 1, 10, 5)
        dust_allergy = st.slider('Dust Allergy Level', 1, 10, 5)
        occupational_hazards = st.slider('Occupational Hazards Level', 1, 10, 5)
        genetic_risk = st.slider('Genetic Risk Level', 1, 10, 5)
        chronic_lung_disease = st.slider('Chronic Lung Disease Level', 1, 10, 5)
        balanced_diet = st.slider('Balanced Diet Level', 1, 10, 5)
        obesity = st.slider('Obesity Level', 1, 10, 5)
        smoking = st.slider('Smoking Level', 1, 10, 5)
        passive_smoker = st.slider('Passive Smoker Level', 1, 10, 5)
        chest_pain = st.slider('Chest Pain Level', 1, 10, 5)
        coughing_of_blood = st.slider('Coughing of Blood Level', 1, 10, 5)
        fatigue = st.slider('Fatigue Level', 1, 10, 5)
        weight_loss = st.slider('Weight Loss Level', 1, 10, 5)
        shortness_of_breath = st.slider('Shortness of Breath Level', 1, 10, 5)
        wheezing = st.slider('Wheezing Level', 1, 10, 5)
        swallowing_difficulty = st.slider('Swallowing Difficulty Level', 1, 10, 5)
        clubbing_of_finger_nails = st.slider('Clubbing of Finger Nails Level', 1, 10, 5)
        frequent_cold = st.slider('Frequent Cold Level', 1, 10, 5)
        dry_cough = st.slider('Dry Cough Level', 1, 10, 5)
        snoring = st.slider('Snoring Level', 1, 10, 5)

    # Convert categorical inputs to numerical
    gender = 1 if gender == 'Female' else 0

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Air Pollution': [air_pollution],
        'Alcohol use': [alcohol_use],
        'Dust Allergy': [dust_allergy],
        'OccuPational Hazards': [occupational_hazards],
        'Genetic Risk': [genetic_risk],
        'chronic Lung Disease': [chronic_lung_disease],
        'Balanced Diet': [balanced_diet],
        'Obesity': [obesity],
        'Smoking': [smoking],
        'Passive Smoker': [passive_smoker],
        'Chest Pain': [chest_pain],
        'Coughing of Blood': [coughing_of_blood],
        'Fatigue': [fatigue],
        'Weight Loss': [weight_loss],
        'Shortness of Breath': [shortness_of_breath],
        'Wheezing': [wheezing],
        'Swallowing Difficulty': [swallowing_difficulty],
        'Clubbing of Finger Nails': [clubbing_of_finger_nails],
        'Frequent Cold': [frequent_cold],
        'Dry Cough': [dry_cough],
        'Snoring': [snoring]
    })

    # Debugging: Print expected and actual columns
    st.write("Expected columns from the model:", expected_columns)
    st.write("Columns in the input data:", input_data.columns.tolist())

    # Ensure columns are in the same order as during model training
    missing_columns = set(expected_columns) - set(input_data.columns)
    if missing_columns:
        st.error(f"The following expected columns are missing in the input data: {missing_columns}")
    else:
        input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            
            st.write(f'Prediction: {"High" if prediction[0] == "high" else "Low"}')
            st.write(f'Probability of High Risk: {probability:.2f}')

            # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))

            # Plot High/Low risk probability
            sns.barplot(x=['Low', 'High'], y=[1 - probability, probability], ax=axes[0], palette=['green', 'red'])
            axes[0].set_title('High/Low Risk Probability')
            axes[0].set_ylabel('Probability')

            # Plot Age distribution
            sns.histplot(input_data['Age'], kde=True, ax=axes[1])
            axes[1].set_title('Age Distribution')

            # Plot High/Low risk pie chart
            axes[2].pie([1 - probability, probability], labels=['Low', 'High'], autopct='%1.1f%%', colors=['green', 'red'])
            axes[2].set_title('High/Low Risk Pie Chart')

            # Display the plots
            st.pyplot(fig)

if __name__ == '__main__':
    main()
