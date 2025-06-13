# IDP – Intelligent Data Prediction for Airfoil Pressure Coefficients

This project focuses on predicting the pressure coefficient (`cp`) on an airfoil surface using advanced machine learning models, including a Neural Network and Gradient Boosting. It is part of a collaborative aerospace and computer science initiative to enhance aerodynamic analysis using data-driven methods.

## 🚀 Project Goal

To design a machine learning-based solution for predicting `cp` values based on parameters:
- X_m: Position along the airfoil chord (0.0 to 1.0)
- AoA_deg: Angle of Attack in degrees (-10° to 30°)
- Pressure_Pa: Static pressure in Pascals

This supports aerodynamic studies by providing accurate interpolations for in-between angles of attack.

## 🧠 Models Used

- **Neural Network (Keras)**: Deep learning model with multiple hidden layers trained for regression on aerodynamic data.
- **Gradient Boosting (sklearn)**: Ensemble-based regression model for comparison and interpretability.

## 📊 Features

- Dynamic web UI for prediction
- Real-time model switching between NN and Gradient Boosting
- Visualizations and model performance metrics
- Historical data integration across multiple Excel sheets
- Integrated with Flask backend and HTML frontend

## 🖥️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Pranav0209/IDP.git
cd IDP
```

### 2. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the App
```bash
python3 app.py
```

Then open your browser and go to `http://127.0.0.1:5000`.

## 📁 File Structure

- `app.py` – Flask backend
- `templates/` – HTML frontend
- `static/` – CSS and saved models
- `Updated_Combined_Dataset_with_Pressure.csv` – Combined dataset from all Excel files
- `.gitignore` – Ensures `venv/` is excluded
- `requirements.txt` – All dependencies

## ⚠️ Notes

- Do **not** track `venv/` in Git (handled via `.gitignore`)
- GitHub max file size is 100MB – keep model files lightweight

