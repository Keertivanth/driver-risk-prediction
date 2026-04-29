from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Simple logic instead of ML model
def calculate_risk(df):
    df = df.select_dtypes(include=[np.number])

    # Risk = average of all values
    risk_score = float(df.mean().mean() * 100)

    return risk_score

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        df = pd.read_csv(file)

        risk = calculate_risk(df)

        driving_type = "Safe" if risk < 50 else "Aggressive"

        return jsonify({
            "type": driving_type,
            "risk": round(risk, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/sample")
def sample():
    try:
        df = pd.read_csv("train_motion_data.csv")

        risk = calculate_risk(df)

        driving_type = "Safe" if risk < 50 else "Aggressive"

        return jsonify({
            "type": driving_type,
            "risk": round(risk, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)