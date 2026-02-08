# app.py

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import json
import os
import warnings

# Pengaturan dan load file
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# load model/scaler/encoder dari folder models/
rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
nb_model = joblib.load(os.path.join(MODELS_DIR, "nb_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
# nama fitur harus sama persis dengan saat training
FEATURE_NAMES = ["bandwidth", "latency", "packet_loss", "uptime"]

app = Flask(__name__)

# Load metrics jika ada (tetap tidak mengubah alur)
metrics_summary = None
metrics_path = os.path.join(BASE_DIR, "metrics_summary.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics_summary = json.load(f)


def get_model(model_name: str):
    """Pilih model berdasarkan input ('rf' atau 'nb')."""
    model_name = (model_name or "rf").lower()
    if model_name == "nb" or model_name == "naive" or model_name == "naïve":
        return nb_model, "Naïve Bayes"
    return rf_model, "Random Forest"


@app.route("/")
def index():
    return render_template("index.html", metrics=metrics_summary)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    # ambil payload (bisa JSON atau form)
    data = request.get_json() if request.is_json else request.form

    try:
        # ambil value dan cast ke float
        bandwidth = float(data.get("bandwidth", 0))
        latency = float(data.get("latency", 0))
        packet_loss = float(data.get("packet_loss", 0))
        uptime = float(data.get("uptime", 0))
        model_choice = data.get("model", "rf")
    except Exception as e:
        return jsonify({"error": "Input tidak valid", "details": str(e)}), 400

    # Buat DataFrame satu baris dengan kolom yang sama seperti saat training
    X_df = pd.DataFrame(
        [[bandwidth, latency, packet_loss, uptime]],
        columns=FEATURE_NAMES
    )

    # Pastikan numeric
    X_df = X_df.astype(float)

    # Terapkan scaler (menggunakan DataFrame menjaga konsistensi nama kolom)
    try:
        features_s = scaler.transform(X_df)
    except Exception as e:
        # fallback: jika scaler menolak, kirim error agar mudah debug
        return jsonify({"error": "Gagal melakukan scaling", "details": str(e)}), 500

    # Pilih model dan melakukan prediksi
    model, model_label = get_model(model_choice)
    try:
        pred_idx = int(model.predict(features_s)[0])
    except Exception as e:
        return jsonify({"error": "Gagal memprediksi", "details": str(e)}), 500

    # Peluang Probabilitas 
    proba = None
    if hasattr(model, "predict_proba"):
        raw_proba = model.predict_proba(features_s)[0]
        # menggunakan label_encoder.classes_ sebagai nama kelas (order sesuai transform)
        try:
            classes = list(label_encoder.classes_)
            # jika panjang kelas tidak sama, fallback ke indeks
            if len(classes) == len(raw_proba):
                proba = {str(c): float(p) for c, p in zip(classes, raw_proba)}
            else:
                proba = {str(i): float(p) for i, p in enumerate(raw_proba)}
        except Exception:
            proba = {str(i): float(p) for i, p in enumerate(raw_proba)}

    # label prediksi (kembalikan nama asli)
    try:
        prediction_label = label_encoder.inverse_transform([pred_idx])[0]
    except Exception:
        prediction_label = str(pred_idx)

    return jsonify({
        "model_used": model_label,
        "prediction": prediction_label,
        "probabilities": proba,
    })


if __name__ == "__main__":
    # debug=True hanya untuk development (sesuaikan kalau ingin di production)
    # Catatan: jika log / warning masih banyak, cek frontend apakah mengirim banyak request berulang
    # atau gunakan warnings.filterwarnings untuk sementara menyembunyikannya.
    app.run(host="127.0.0.1", port=5001, debug=True)


