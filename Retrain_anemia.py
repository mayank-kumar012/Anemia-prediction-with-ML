import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# =====================================================
# 1️⃣ Load your dataset
# =====================================================
# Replace with your dataset file
data = pd.read_csv("anemia data from Kaggle.csv")

# Expected columns:
# ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV', 'Result']
# where 'Result' is 1 = Anemic, 0 = Non-Anemic

print("Dataset loaded successfully. Shape:", data.shape)
print("Columns:", data.columns.tolist())

# =====================================================
# 2️⃣ Basic preprocessing
# =====================================================
# See what the original values look like

# Convert gender to numeric (Male = 0, Female = 1)
data["Gender"] = data["Gender"].astype(int)
# Handle missing values (if any)
data = data.dropna(subset=["Hemoglobin", "MCV", "Result"])

# Define features and target
X = data[["Gender", "Hemoglobin", "MCV"]]
y = data["Result"]


print("\n--- DEBUG INFO ---")

print("Columns in dataset:", data.columns.tolist())
print("Unique values in Gender column:", data["Gender"].unique())
print("Number of missing values per column:\n", data.isna().sum())
print("Remaining rows after cleaning:", len(data))

print("\nSample of dataset:")
print(data.head())

print("\nShape of features (X) and target (y):")
print(X.shape, y.shape)


# =====================================================
# 3️⃣ Split into training and test sets
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# 4️⃣ Build a Pipeline (Scaler + RandomForest)
# =====================================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    ))
])

# =====================================================
# 5️⃣ Train the model
# =====================================================
pipeline.fit(X_train, y_train)

# =====================================================
# 6️⃣ Evaluate model performance
# =====================================================
y_pred = pipeline.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =====================================================
# 7️⃣ Save the pipeline and model separately
# =====================================================
# Full pipeline (preprocessor + classifier)
joblib.dump(pipeline, "pipeline_random_forest_retrained.pkl")

# Classifier only
# joblib.dump(pipeline.named_steps["rf"], "random_forest_model_retrained.pkl")

print("\n✅ Model retrained and saved successfully!")
print("Saved files:")
print("- pipeline_random_forest_retrained.pkl")
# print("- random_forest_model_retrained.pkl")
