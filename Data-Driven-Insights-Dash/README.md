# 📊 Data-Driven Insights Dashboard

A Machine Learning-powered Streamlit dashboard for **Customer Churn Prediction** using Random Forest.

---

## 🚀 Features
- Interactive UI built with Streamlit
- Real-time customer churn prediction
- Confusion matrix visualization
- Modular code (train + app separation)
- Deployable on Streamlit Cloud

---

## 🧠 Tech Stack
- Python 3.9+
- Pandas, NumPy, Scikit-learn
- Streamlit
- Seaborn, Matplotlib

---

## ⚙️ How to Run Locally
```bash
# 1. Create and activate venv
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python src/train_model.py

# 4. Launch Streamlit dashboard
streamlit run src/app.py
