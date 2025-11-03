import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import load_data, split_data, scale_data

# Step 1: Load dataset
data_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = load_data(data_path)

# Step 2: Split data
X_train, X_test, y_train, y_test = split_data(df, target_col="Churn")

# Step 3: Scale data
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

# Step 4: Train RandomForest model
model = RandomForestClassifier(random_state=42, n_estimators=120)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Step 5: Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Confusion matrix visualization
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("assets/confusion_matrix.png")
plt.close()

# Step 7: Save model and scaler

joblib.dump({"model": model, "scaler": scaler}, "../assets/pipeline.joblib")

print("✅ Model training complete! Saved to assets/pipeline.joblib")
