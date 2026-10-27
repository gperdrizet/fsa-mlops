'''Model training for Wine classification using Random Forest'''

from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
data = load_wine()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
current_dir = Path(__file__).resolve().parent
os.makedirs(current_dir / 'models', exist_ok=True)
joblib.dump(model, current_dir / 'models/wine_model.pkl')
print("Model training complete and saved.")
