import pickle
from flask import Flask, request, app, jsonify, render_template, url_for
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
#Load the model
churn_model = pickle.load(open('churn_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
explainer = pickle.load(open('explainer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    df = pd.DataFrame([data])
    # 1. Transform the categorical 'salary' using encoder.pkl
    salary_encoded = encoder.transform(df[['salary']])
    salary_df = pd.DataFrame(
        salary_encoded, 
        columns=encoder.get_feature_names_out(['salary'])
    )
    # 2. Drop the string 'salary' and concat the numeric dummies
    # Ensure empid is dropped bacause it wasn't used in training
    df_numeric = df.drop(['salary', 'empid'], axis=1, errors='ignore')
    final_df = pd.concat([df_numeric, salary_df], axis=1)
    
    output = churn_model.predict(final_df)
    return jsonify(int(output[0]))

@app.route('/predict', methods=['POST'])
def predict():
    # Capture form data as a dictionary
    # This gets values from <input name="salary">, <input name="satisfaction_level">, etc.
    form_data = request.form.to_dict()
    df = pd.DataFrame([form_data])
     # Convert numeric strings to actual numbers
    numeric_features = [
        'satisfaction_level', 'last_evaluation', 'number_project', 
        'average_montly_hours', 'time_spend_company', 
        'Work_accident', 'promotion_last_5years'
    ]
    # Convert only these to float; 'salary' stays as a string
    df[numeric_features] = df[numeric_features].astype(float)
     # Use the Encoder for 'salary'
    salary_encoded = encoder.transform(df[['salary']])
    salary_df = pd.DataFrame(
        salary_encoded, 
        columns=encoder.get_feature_names_out(['salary'])
    )
    # Combine numeric data and encoded salary columns
    # We ignore 'empid' and 'salary' (the string version)
    final_input = pd.concat([df[numeric_features], salary_df], axis=1)
    prediction = churn_model.predict(final_input)[0]

    shap_values = explainer(final_input)
    plt.figure(figsize=(10,6))
    shap.plots.waterfall(shap_values[0], show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()

    plot_url = base64.b64encode(buf.getbuffer()).decode("utf-8")

    if prediction == 1:
        display_text = "Employee is likely to LEAVE."
    else:
        display_text = "Employee is likely to STAY."

    #shap_vaues = explainer(final_input)
    return render_template('home.html', prediction_text=f'Employee Churn Prediction: {display_text}',shap_plot=plot_url)

if __name__ == "__main__":
    app.run(debug=True)