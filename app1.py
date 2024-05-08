import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap

# Load the RandomForestRegressor model and SHAP explainer
with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('explainer.pkl', 'rb') as file:
    explainer = pickle.load(file)

# Define function to generate SHAP plot as HTML
def generate_shap_plot(explainer, input_features_array, features_df):
    shap_values = explainer.shap_values(input_features_array)
    shap_plot_html = shap.force_plot(explainer.expected_value, shap_values, features_df)
    return shap_plot_html

# Define function to make prediction
def predict_score(input_features_array):
    return rf_model.predict(input_features_array)[0]

# Streamlit app
def main():
    st.title('Student Performance Prediction')

    # Collect user inputs for non-one-hot encoded features
    age = st.number_input('Age', min_value=0, max_value=100, value=15)
    absences = st.number_input('Absences', min_value=0, max_value=20, value=10)
    G1 = st.number_input('First Period Grade (G1)', min_value=0, max_value=20, value=10)
    G2 = st.number_input('Second Period Grade (G2)', min_value=0, max_value=20, value=10)

    # Create dictionary to store features
    features = {
        'age': age,
        'absences': absences,
        'G1': G1,
        'G2': G2
    }

    # Define one-hot encoded attributes
    one_hot_attributes = {
        'sex': ['sex_F', 'sex_M'],
        'address': ['address_R', 'address_U'],
        'Medu': ['Medu_0', 'Medu_1', 'Medu_2', 'Medu_3', 'Medu_4'],
        'Fedu': ['Fedu_0','Fedu_1', 'Fedu_2', 'Fedu_3', 'Fedu_4'],
        'health': ['health_1', 'health_2', 'health_3', 'health_4', 'health_5'],
        'freetime': ['freetime_1', 'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5'],
        'goout': ['goout_1', 'goout_2', 'goout_3', 'goout_4', 'goout_5'],
        'studytime': ['studytime_1', 'studytime_2', 'studytime_3', 'studytime_4'],
        'failures': ['failures_0', 'failures_1', 'failures_2', 'failures_3'],
        'famsize': ['famsize_GT3', 'famsize_LE3'],
        'Pstatus': ['Pstatus_A', 'Pstatus_T'],
        'Mjob': ['Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher'],
        'Fjob': ['Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher'],
        'guardian': ['guardian_father', 'guardian_mother', 'guardian_other'],
        'famsup': ['famsup_no', 'famsup_yes'],
        'paid': ['paid_no', 'paid_yes'],
        'activities': ['activities_no', 'activities_yes'],
        'higher': ['higher_no', 'higher_yes'],
        'internet': ['internet_no', 'internet_yes']
    }

    # Collect user inputs for one-hot encoded attributes
    for attribute, options in one_hot_attributes.items():
        selected_option = st.selectbox(attribute, options)
        for option in options:
            features[option] = option == selected_option

    # Convert features to DataFrame and then to NumPy array
    features_df = pd.DataFrame(features, index=[0])
    input_features_array = np.array(features_df)

    # Perform prediction
    predicted_score = predict_score(input_features_array)

    # Generate SHAP plot as HTML
    shap_plot_html = generate_shap_plot(explainer, input_features_array, features_df)

    # Display prediction and SHAP plot
    st.subheader('Prediction')
    st.write(f'The predicted score is: {predicted_score}')

    st.subheader('SHAP Plot')
    st.write(shap_plot_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()