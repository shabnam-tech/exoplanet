from flask import Flask, request, jsonify
import sqlite3
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("habitability_model.pkl")
features = joblib.load("model_features.pkl")

def get_db():
    return sqlite3.connect("exoplanets.db")

# ADD PLANET ENDPOINT
@app.route("/add_planet", methods=["POST"])
def add_planet():
    data = request.json

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO planets (
        pl_rade, pl_bmasse, pl_orbsmax,
        st_teff, st_met, st_luminosity,
        pl_luminosity, stellar_compatibility_index
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["pl_rade"],
        data["pl_bmasse"],
        data["pl_orbsmax"],
        data["st_teff"],
        data["st_met"],
        data["st_luminosity"],
        data["pl_luminosity"],
        data["stellar_compatibility_index"]
    ))

    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "Planet added successfully"})

# PREDICT SINGLE PLANET
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_df = pd.DataFrame([data])
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return jsonify({
        "habitability_class": "Habitable" if prediction == 1 else "Non-Habitable",
        "habitability_probability": round(float(probability), 3)
    })

# RANK ALL PLANETS
@app.route("/rank", methods=["GET"])
def rank_planets():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM planets", conn)
    conn.close()

    if df.empty:
        return jsonify({"message": "No planets found in database"})

    X = df[features]
    df["habitability_score"] = model.predict_proba(X)[:, 1]

    ranked = df.sort_values("habitability_score", ascending=False)

    return jsonify(
        ranked[["id", "habitability_score"]].to_dict(orient="records")
    )

if __name__ == "__main__":
    app.run(debug=True)
