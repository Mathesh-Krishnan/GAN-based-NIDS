import os
import smtplib
from flask import Flask, render_template, request, jsonify
import xgboost as xgb
import pandas as pd
import google.generativeai as genai
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure the Gemini API
genai.configure(api_key="API Key")

# Initialize the Gemini model
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load the XGBoost model
model = xgb.XGBClassifier()
model.load_model('xgboost_model_reduced.json')
print("Loaded XGBoost model from 'xgboost_model_reduced.json'.")

# Define the feature names used in the XGBoost model
features = ['Machine', 'DebugSize', 'DebugRVA', 'MajorImageVersion', 'MajorOSVersion',
            'ExportRVA', 'ExportSize', 'IatVRA', 'MajorLinkerVersion', 'MinorLinkerVersion',
            'NumberOfSections', 'SizeOfStackReserve', 'DllCharacteristics', 'ResourceSize',
            'BitcoinAddresses']

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_FROM = os.getenv('EMAIL_FROM')  
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')  
EMAIL_TO = 'bashith67@gmail.com' 

def send_email(prediction, confidence, gemini_response):
    """Send an email with the prediction results."""
    if not EMAIL_FROM or not EMAIL_PASSWORD:
        print("Error: Email credentials not set in environment variables.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = "Prediction Results"

        body = f"""
        Model Prediction: {prediction}
        Confidence: {confidence * 100:.2f}% (if available)
        AI Analysis: {gemini_response}
        """
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()

        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Create a sample data dictionary using form input for all features
        sample_data = {}
        for feature in features:
            value = request.form.get(feature)
            if value is None:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            try:
                sample_data[feature] = float(value)  # Convert to float to handle numeric inputs
            except ValueError:
                return jsonify({'error': f'Invalid value for {feature}: {value}'}), 400

        sample_df = pd.DataFrame([sample_data])

        # Make a prediction
        prediction = model.predict(sample_df)[0]
        probabilities = model.predict_proba(sample_df)[0]
        confidence = max(probabilities)

        # Map the prediction (0 = Benign, 1 = Malicious)
        label_mapping = {0: "Benign", 1: "Malicious"}
        predicted_label = label_mapping.get(prediction, "Unknown")

        # Get AI analysis using Gemini-1.5-flash
        input_text = f"The model predicted {predicted_label} with {confidence * 100:.2f}% confidence. Provide insights."
        try:
            response = gemini_model.generate_content(input_text)
            gemini_response = response.text
        except Exception as e:
            gemini_response = f"Error generating response from Gemini: {str(e)}"

        # Send email notification with the prediction and AI analysis
        print("About to send email...")
        send_email(predicted_label, confidence, gemini_response)
        print("Email function completed.")

        return render_template('result.html', prediction=predicted_label, confidence=confidence, gpt_response=gemini_response)

    return render_template('predict.html', features=features)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']

        # Prepare input for Gemini with a focus on cyber network solutions
        chatbot_input = f"User: {user_input}\nChatbot: Please provide information and solutions related to AI-ENHANCED INTRUSION DETECTION SYSTEM."

        try:
            chatbot_response = gemini_model.generate_content(chatbot_input)
            gemini_response = chatbot_response.text
        except Exception as e:
            gemini_response = f"Error generating response from Gemini: {str(e)}"

        return jsonify({'response': gemini_response})

    return render_template('chat.html')

if __name__ == '__main__':

    app.run(debug=True)
