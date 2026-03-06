"""
Stellar Analytics - Flask Backend
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

try:
    clf_model = joblib.load(os.path.join(BASE, "models", "best_classifier.pkl"))
    reg_model = joblib.load(os.path.join(BASE, "models", "best_regressor.pkl"))
    feat_A    = joblib.load(os.path.join(BASE, "models", "top_features_A.pkl"))
    feat_B    = joblib.load(os.path.join(BASE, "models", "top_features_B.pkl"))
    scaler_A  = joblib.load(os.path.join(BASE, "models", "scaler_A.pkl"))
    scaler_B  = joblib.load(os.path.join(BASE, "models", "scaler_B.pkl"))
    imputer_A = joblib.load(os.path.join(BASE, "models", "imputer_A.pkl"))
    imputer_B = joblib.load(os.path.join(BASE, "models", "imputer_B.pkl"))

    # These are the FULL feature lists the scaler/imputer were fitted on
    SCALER_FEATURES_A = list(scaler_A.feature_names_in_)
    SCALER_FEATURES_B = list(scaler_B.feature_names_in_)

    print("Models loaded!")
    print("Scaler A features:", SCALER_FEATURES_A)
    print("Scaler B features:", SCALER_FEATURES_B)
    print("Model A features:", feat_A)
    print("Model B features:", feat_B)

except Exception as e:
    print(f"Model loading error: {e}")
    clf_model = reg_model = None
    SCALER_FEATURES_A = []
    SCALER_FEATURES_B = []

prediction_history = []

FIELD_RANGES = {
    "koi_period":       (0.5,   1000.0,  "Orbital period in days"),
    "koi_duration":     (0.5,   24.0,    "Transit duration in hours"),
    "koi_depth":        (10.0,  100000.0,"Transit depth in ppm"),
    "koi_impact":       (0.0,   1.5,     "Impact parameter (0-1.5)"),
    "koi_model_snr":    (0.0,   2000.0,  "Signal-to-noise ratio"),
    "koi_num_transits": (1.0,   5000.0,  "Number of observed transits"),
    "koi_ror":          (0.001, 0.9,     "Ratio of planet to star radius"),
    "teff":             (2500.0,10000.0, "Stellar temperature in Kelvin"),
    "logg":             (1.0,   5.5,     "Stellar surface gravity (log scale)"),
    "feh":              (-2.5,  1.0,     "Stellar metallicity [Fe/H]"),
}

def validate_input(data):
    errors = {}
    cleaned = {}
    for field, (min_val, max_val, desc) in FIELD_RANGES.items():
        val = data.get(field)
        if val is None or str(val).strip() == "":
            errors[field] = f"{field} is required."
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            errors[field] = f"{field} must be a number."
            continue
        if not (min_val <= val <= max_val):
            errors[field] = f"{field} must be between {min_val} and {max_val}."
            continue
        cleaned[field] = val
    return cleaned, errors


def build_all_features(cleaned):
    """
    Build every possible feature from raw user input.
    This must cover ALL features the scaler was fitted on.
    Scaler A features: koi_period, koi_duration, koi_depth, koi_impact,
                       koi_model_snr, koi_num_transits, koi_ror,
                       teff_uncertainty, logg_uncertainty, feh_uncertainty,
                       depth_per_period, snr_per_transit
    """
    d = cleaned.copy()

    # Engineered features
    d["teff_uncertainty"] = 0.0
    d["logg_uncertainty"] = 0.0
    d["feh_uncertainty"]  = 0.0
    d["depth_per_period"] = d["koi_depth"]     / (d["koi_period"]       + 1e-9)
    d["snr_per_transit"]  = d["koi_model_snr"] / (d["koi_num_transits"] + 1e-9)

    return d


def preprocess(cleaned, scaler, imputer, scaler_features, model_features):
    """
    Full preprocessing pipeline:
    1. Build all features
    2. Align to scaler feature order
    3. Impute missing values
    4. Scale
    5. Select only the features the model needs
    """
    # Step 1 - build all features
    all_feats = build_all_features(cleaned)
    df = pd.DataFrame([all_feats])

    # Step 2 - align to scaler feature order (fills missing with NaN)
    df = df.reindex(columns=scaler_features, fill_value=np.nan)

    # Step 3 - impute
    df_imp = imputer.transform(df)
    df_imp = pd.DataFrame(df_imp, columns=scaler_features)

    # Step 4 - scale
    df_scaled = scaler.transform(df_imp)
    df_scaled = pd.DataFrame(df_scaled, columns=scaler_features)

    # Step 5 - select only model features
    df_final = df_scaled[model_features]

    return df_final


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Stellar Analytics API",
        "status":  "running",
        "feat_A":  feat_A if clf_model else [],
        "feat_B":  feat_B if reg_model else []
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "models_loaded": clf_model is not None,
        "feat_A":        feat_A if clf_model else [],
        "feat_B":        feat_B if reg_model else [],
        "timestamp":     datetime.utcnow().isoformat()
    })


@app.route("/fields", methods=["GET"])
def get_fields():
    fields = []
    for name, (min_v, max_v, desc) in FIELD_RANGES.items():
        fields.append({
            "name":        name,
            "label":       name.replace("koi_", "").replace("_", " ").title(),
            "description": desc,
            "min":         min_v,
            "max":         max_v,
        })
    return jsonify({"fields": fields})


@app.route("/predict/full", methods=["POST"])
def predict_full():
    if clf_model is None or reg_model is None:
        return jsonify({"error": "Models not loaded."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    cleaned, errors = validate_input(data)
    if errors:
        return jsonify({"errors": errors}), 422

    try:
        # ── Task A ──────────────────────────────────────────────
        df_a = preprocess(
            cleaned,
            scaler_A, imputer_A,
            SCALER_FEATURES_A,
            feat_A
        )
        pred_class = int(clf_model.predict(df_a)[0])
        pred_proba = clf_model.predict_proba(df_a)[0].tolist()
        label_map  = {1: "CONFIRMED", 0: "FALSE POSITIVE"}

        classification = {
            "prediction":  label_map[pred_class],
            "label":       pred_class,
            "confidence":  round(max(pred_proba) * 100, 2),
            "probabilities": {
                "CONFIRMED":      round(pred_proba[1] * 100, 2),
                "FALSE_POSITIVE": round(pred_proba[0] * 100, 2)
            }
        }

        # ── Task B (only if CONFIRMED) ───────────────────────────
        regression = None
        if pred_class == 1:
            df_b = preprocess(
                cleaned,
                scaler_B, imputer_B,
                SCALER_FEATURES_B,
                feat_B
            )
            pred_log    = float(reg_model.predict(df_b)[0])
            pred_radius = float(np.expm1(pred_log))

            if pred_radius < 1.25:
                category = "Rocky / Earth-like"
            elif pred_radius < 2.0:
                category = "Super-Earth"
            elif pred_radius < 4.0:
                category = "Mini-Neptune"
            elif pred_radius < 10.0:
                category = "Neptune-like"
            else:
                category = "Gas Giant / Jupiter-like"

            regression = {
                "predicted_radius": round(pred_radius, 4),
                "unit":             "Earth radii",
                "planet_category":  category
            }

        result = {
            "classification": classification,
            "regression":     regression,
            "input":          cleaned,
            "timestamp":      datetime.utcnow().isoformat()
        }

        prediction_history.append(result)
        return jsonify(result), 200

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


@app.route("/predict/classify", methods=["POST"])
def predict_classify():
    return predict_full()


@app.route("/predict/radius", methods=["POST"])
def predict_radius():
    return predict_full()


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify({
        "count":   len(prediction_history),
        "history": prediction_history[-20:]
    })


@app.route("/history", methods=["DELETE"])
def clear_history():
    prediction_history.clear()
    return jsonify({"message": "History cleared."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

@app.route('/')
def index():
    return app.send_static_file('index.html')