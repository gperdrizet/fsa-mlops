'''Model training script for GitHub actions demo'''

from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=150)
clf.fit(X_train, y_train)

# Save model
current_dir = Path(__file__).resolve().parent
model_path = current_dir / "iris_model.pkl"
joblib.dump(clf, model_path)
print(f" Model trained and saved as {model_path}")
