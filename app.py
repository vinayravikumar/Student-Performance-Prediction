import uuid
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import os 

# Load the RandomForestRegressor model and SHAP explainer
with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)
    
with open('explainer.pkl', 'rb') as file:
    explainer = pickle.load(file)

app = Flask(__name__)
SAVE_FOLDER = 'static/images/'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect user inputs for all features
    features = {}

    # Add other non-one-hot encoded features from the form data
    features['age'] = float(request.form.get('age'))
    features['absences'] = float(request.form.get('absences'))
    features['G1'] = float(request.form.get('G1'))
    features['G2'] = float(request.form.get('G2'))

    # Get the form data for all one-hot encoded attributes
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

    for attribute, options in one_hot_attributes.items():
        selected_option = request.form.get(attribute)
        for option in options:
            features[option] = option == selected_option

    # Convert dictionary to DataFrame
    features_df = pd.DataFrame.from_dict(features, orient='index').T

    # Convert DataFrame to NumPy array
    input_features_array = np.array(features_df)

    # Perform prediction
    predicted_score = rf_model.predict(input_features_array)[0]
    
    shap_values = explainer.shap_values(input_features_array)

    plot_id = str(uuid.uuid4())
    
    # Save the SHAP summary plot with the unique identifier
    shap_plot_file = os.path.join(SAVE_FOLDER, f'shap_summary_plot_{plot_id}.png')
    shap.summary_plot(shap_values, features_df, show=False)
    plt.savefig(shap_plot_file)
    plt.close()

    # Render result.html with predicted score and SHAP plot file path
    return render_template('result.html', predicted_score=predicted_score, shap_plot_file=shap_plot_file)

if __name__ == '__main__':
    app.run(debug=True)
