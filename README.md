# Data-Driven-Insights-Dashboard
An interactive data analytics dashboard built using Python, Pandas, and Plotly.
# 📊 Customer Churn Prediction Dashboard

An end-to-end Machine Learning project that predicts customer churn using a Random Forest model and presents insights through an interactive Streamlit dashboard.

---

## 🚀 Overview

This project focuses on analyzing customer data to identify patterns that lead to churn and building a predictive model to classify whether a customer is likely to leave.

The solution combines data preprocessing, exploratory data analysis (EDA), machine learning, and interactive visualization.

---

## 🧠 Key Features

- 📌 Data Cleaning and Preprocessing  
- 📊 Exploratory Data Analysis (EDA)  
- 🤖 Customer Churn Prediction using Random Forest  
- 📈 Model Evaluation (Confusion Matrix, Accuracy, Precision, Recall)  
- 🖥️ Interactive Dashboard built with Streamlit  
- ⚡ Real-time prediction based on user input  

---

## 🛠️ Tech Stack

- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib  
- **Framework:** Streamlit  
- **Tools:** Jupyter Notebook, Git  

---

## 📂 Project Structure
Data-Driven-Insights-Dashboard/
│── assets/ # Images / model / confusion matrix
│── data/ # Customer churn dataset
│── src/
│ ├── train_model.py # Model training script
│ ├── app.py # Streamlit dashboard
│── requirements.txt # Dependencies
│── README.md # Project documentation


---

## ⚙️ How It Works

1. Data is cleaned and preprocessed  
2. EDA is performed to understand trends and patterns  
3. A Random Forest model is trained on the dataset  
4. The model is integrated into a Streamlit app  
5. Users can input data and get real-time churn predictions  

---

## ▶️ Run Locally

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python src/train_model.py

# Run dashboard
streamlit run src/app.py
