from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Evaluate and print accuracy
accuracy = accuracy_score(y_test, test_preds)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
current_dir = Path(__file__).resolve().parent
joblib.dump(model, f"{current_dir}/model.pkl")

# Save datasets with target and predictions explicitly
train_df = pd.DataFrame(X_train, columns=data.feature_names)
train_df['target'] = y_train
train_df['prediction'] = train_preds
train_df.to_csv(f"{current_dir}/train_data.csv", index=False)

test_df = pd.DataFrame(X_test, columns=data.feature_names)
test_df['target'] = y_test
test_df['prediction'] = test_preds
test_df.to_csv(f"{current_dir}/reference_data.csv", index=False)
