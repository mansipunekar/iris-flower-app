# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ---------------------------
# Train and Save Model
# ---------------------------
def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open("iris_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Train model once
train_model()

# Load model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

iris = load_iris()

# ---------------------------
# Flask Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(data)[0]
        species = iris.target_names[prediction]

        return render_template("index.html", prediction_text=f"Predicted Species: {species}")

if __name__ == "__main__":
    app.run(debug=True)
