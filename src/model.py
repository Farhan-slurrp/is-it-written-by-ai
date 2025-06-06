import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_baseline_model(X_train, y_train):
    """
    Train the baseline model with logistic regression
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)

    print("Classification Report:\n")
    print(classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def save_model(model, path):

    path = os.path.join('..', path)

    joblib.dump(model, path)
    print(f"Model saved to: {path}")
