# Flask app to serve interactive cp visualizations

from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from tensorflow import keras
from flask_cors import CORS

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)
CORS(app)

# Load your final dataset
DATA_PATH = "static/data/dataset.csv"  # Place your CSV here

df = pd.read_csv(DATA_PATH)

# Train model if not already trained
features = ["X_m", "AoA_deg", "Pressure_Pa"]
target = "cp"
X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = Sequential([
    keras.Input(shape=(X_train.shape[1],)),  # Use keras.Input for Keras 3.x compatibility
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)

# Optionally evaluate the model
y_pred = model.predict(X_test).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"✅ RMSE: {rmse:.3f}")
print(f"✅ R² Score: {r2:.3f}")

# Train Gradient Boosting model for comparison
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)
print(f"✅ [GB] RMSE: {rmse_gb:.3f}")
print(f"✅ [GB] R² Score: {r2_gb:.3f}")

# Save the trained model to disk
model.save('static/model/model.keras')

# Load Keras model instead of pickle
model = keras.models.load_model('static/model/model.keras', compile=False)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/prediction")
def prediction():
    # Explicitly define available angles of attack
    aoa_values = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    explanation = {
        "aoa": "Angle of attack (degrees). Determines the orientation of the airfoil relative to airflow.",
        "x_m": "Position along the airfoil chord. Typically between 0 (leading edge) and 1 (trailing edge).",
        "pressure": "Local static pressure in Pascals. Input to better estimate cp using the trained model."
    }
    return render_template("prediction.html", aoa_values=aoa_values, explanation=explanation)

@app.route("/get_cp_data")
def get_cp_data():
    aoa = request.args.get("aoa")
    if aoa is None or aoa == "":
        return jsonify({"error": "Missing 'aoa' query parameter."}), 400

    try:
        aoa = float(aoa)
    except ValueError:
        return jsonify({"error": "Invalid 'aoa' value."}), 400

    # Use tolerance for float comparison
    aoa_col = df["AoA_deg"].astype(float)
    subset = df[(aoa_col - aoa).abs() < 1e-6][["X_m", "cp"]].dropna()

    if subset.empty:
        return jsonify({"error": f"No data found for AoA = {aoa}"}), 404

    return jsonify(subset.to_dict(orient="records"))

@app.route("/predict_cp")
def predict_cp():
    try:
        x_m = float(request.args.get("x_m", ""))
        aoa = float(request.args.get("aoa", ""))
        pressure = float(request.args.get("pressure", ""))
    except ValueError:
        return jsonify({"error": "Invalid input values."}), 400

    # Validate input ranges
    if not (0.0 <= x_m <= 1.0):
        return jsonify({"error": "x_m must be between 0 and 1."}), 400
    if not (-20 <= aoa <= 40):  # Adjust based on actual training data range
        return jsonify({"error": "AoA must be between -20 and 40 degrees."}), 400
    if not (50000 <= pressure <= 200000):  # Approximate realistic pressure range in Pascals
        return jsonify({"error": "Pressure must be between 50,000 and 200,000 Pa."}), 400

    input_data = [[x_m, aoa, pressure]]
    input_scaled = scaler.transform(input_data)

    print(f"Input received: x_m={x_m}, aoa={aoa}, pressure={pressure}")
    print(f"Scaled input: {input_scaled}")

    nn_prediction = model.predict(input_scaled)[0][0]
    gb_prediction = gb_model.predict(input_scaled)[0]

    print(f"NN cp: {nn_prediction}, GB cp: {gb_prediction}")

    return jsonify({
        "nn_predicted_cp": float(nn_prediction),
        "gb_predicted_cp": float(gb_prediction)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
