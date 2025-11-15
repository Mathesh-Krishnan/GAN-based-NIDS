**🛡️ GAN-Based Network Intrusion Detection System**

A hybrid GAN + Machine Learning intrusion detection system, combined with a Flask-based web interface for real-time prediction and anomaly detection.

**📌 Overview**

This project implements a Generative Adversarial Network (GAN) to enhance intrusion detection by generating synthetic malicious traffic and improving classifier performance.

It consists of two main components:

**1️⃣ TRAIN_CODE**

Jupyter notebooks, training scripts, EDA files, and model artifacts.

Implements GAN training, synthetic data generation, classifier training, and evaluation.

**2️⃣ WEBSITE**

Flask-based web application with HTML/CSS frontend.

Allows users to upload or input network parameters and get predictions using the trained model.

The system is designed for academic/final-year research work demonstrating how GANs can strengthen network security models.

** 🚀 Features
🧠 Machine Learning + GAN**

Trains a Generator to produce synthetic attack samples

Trains a Discriminator to distinguish normal vs malicious

Augments real dataset with generated data

Improves classifier accuracy on imbalanced datasets

Final classifier trained using:

Random Forest

XGBoost (optional)

**🌐 Web Application**

1) Flask API for prediction
2) HTML-based frontend (inside templates/)
3) Takes user input parameters
4) Returns classification:
5) Normal Traffic
6) Malicious / Attack Detected

**📊 Evaluation Tools**

1) Confusion Matrix
2) Classification Report
3) Visualizations via Seaborn + Matplotlib

**🛠️ Tech Stack**

1) Machine Learning
2) Python
3) TensorFlow / Keras
4) Scikit-Learn
5) Pandas / NumPy
6) Matplotlib / Seaborn
7) XGBoost
8) Web Application
9) Flask
10) HTML / CSS / JS

**🧪 Training the Model (TRAIN_CODE)**

Open Jupyter notebooks:
  jupyter notebook


**Run the following in order:**

  1) EDA.ipynb → Preprocessing & Exploration
  
  2) gan_data.ipynb → GAN Training
  
  3) train_code.ipynb → Classifier Training
  
  4) gan_test.ipynb → Model Evaluation
  
  5) Model files such as:
  
  6) best_model.pkl
  
  7) xgboost_model_reduced.json


 ** 🌐 Running the Web Application (WEBSITE)**

Navigate to the website directory:

  1) cd WEBSITE
  2) Run the Flask server:
  3) python app.py
  4) Open browser:
  5) http://127.0.0.1:5000/
  6) You can now:
  7) Enter network parameters
  8) Submit the form
  View real-time detection results

**🎓 Academic Value**

This project demonstrates:

GANs for network intrusion detection

Handling imbalanced network traffic datasets

Synthetic data augmentation

Hybrid ML + Deep Learning design

Model deployment using Flask

Web-based cybersecurity dashboards

**Perfect for final year engineering projects, cybersecurity portfolios, and research papers.**

**🤝 Contributing**

Pull requests are welcome.
For major changes, please open an issue first to discuss what you'd like to modify.

**📜 License**

This project is for academic use only.
You may reuse code with proper credit.
  
