# train_models.py

import os
import json
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)

import joblib
import matplotlib.pyplot as plt

# Menentukan folder dasar proyek dan folder penyimpanan grafik/model
BASE_DIR = os.path.dirname(__file__)

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Memuat dataset dari file CSV yang di arahkan ke base directory
data_path = os.path.join(BASE_DIR, "network_data.csv")
df = pd.read_csv(data_path)

# Menentukan fitur input dan label output
FEATURE_NAMES = ["bandwidth", "latency", "packet_loss", "uptime"]
X = df[FEATURE_NAMES]
y = df["label"]

# 2. Mengubah label teks (normal/gangguan) menjadi angka
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 3. Membagi dataset menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

# 4. Normalisasi fitur numerik agar skala nilai lebih seimbang
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 5. menginisialisasi dua model machine learning (Random Forest dan Naive Bayes)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
nb = GaussianNB()

# 6. Melatih model menggunakan data training
rf.fit(X_train_s, y_train)
nb.fit(X_train_s, y_train)

# 7. Fungsi untuk mengevaluasi model dan menghasilkan metrik
def eval_model(model, X_te, y_te, name):
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    cm = confusion_matrix(y_te, y_pred)

    print(f"\n== HASIL MODEL {name} ==")
    print(f"Akurasi   : {acc:.4f}")
    print(f"Presisi   : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_te, y_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", cm)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "cm": cm,
    }

# Menghitung metrik evaluasi untuk kedua model algoritma yang saya gunakan
rf_metrics = eval_model(rf, X_test_s, y_test, "Random Forest")
nb_metrics = eval_model(nb, X_test_s, y_test, "Naïve Bayes")

# 8. Perhitungan pada probabilitas dengan kurva ROC/PR
try:
    gangguan_index = int(le.transform(["gangguan"])[0])
except Exception:
    print("Peringatan: label 'gangguan' tidak ditemukan:", le.classes_)
    gangguan_index = 1 if 1 in np.unique(y_enc) else 0

rf_proba = rf.predict_proba(X_test_s)[:, gangguan_index]
nb_proba = nb.predict_proba(X_test_s)[:, gangguan_index]

y_test_bin = (y_test == gangguan_index).astype(int)

fpr_rf, tpr_rf, _ = roc_curve(y_test_bin, rf_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_nb, tpr_nb, _ = roc_curve(y_test_bin, nb_proba)
roc_auc_nb = auc(fpr_nb, tpr_nb)

prec_rf, rec_rf, _ = precision_recall_curve(y_test_bin, rf_proba)
pr_auc_rf = auc(rec_rf, prec_rf)

prec_nb, rec_nb, _ = precision_recall_curve(y_test_bin, nb_proba)
pr_auc_nb = auc(rec_nb, prec_nb)

# 9. Menyimpan model, scaler, dan label encoder ke folder models/
joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(nb, os.path.join(MODELS_DIR, "nb_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

# 10. Penyimpanan metrik evaluasi kedalam format JSON untuk ditampilkan di web
metrics_summary = {
    "rf": {
        "accuracy": rf_metrics["accuracy"],
        "precision": rf_metrics["precision"],
        "recall": rf_metrics["recall"],
        "f1": rf_metrics["f1"],
        "roc_auc": float(roc_auc_rf),
        "pr_auc": float(pr_auc_rf),
    },
    "nb": {
        "accuracy": nb_metrics["accuracy"],
        "precision": nb_metrics["precision"],
        "recall": nb_metrics["recall"],
        "f1": nb_metrics["f1"],
        "roc_auc": float(roc_auc_nb),
        "pr_auc": float(pr_auc_nb),
    },
    "label_classes": list(le.classes_),
    "feature_names": FEATURE_NAMES,
}

with open(os.path.join(BASE_DIR, "metrics_summary.json"), "w") as f:
    json.dump(metrics_summary, f, indent=2)

# 11. Membuat untuk gambar confusion matrix yang nantinya akan menyimpannya di folder static
def plot_confusion(cm, model_name, filename):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    classes = le.classes_
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, filename))
    plt.close(fig)

plot_confusion(rf_metrics["cm"], "Random Forest", "rf_confusion.png")
plot_confusion(nb_metrics["cm"], "Naïve Bayes", "nb_confusion.png")

# 12. : EDA (Exploratory Data Analysis) + Visualisasi

print("Distribusi Label:")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True).round(3))

fitur_numerik = ["bandwidth", "latency", "packet_loss", "uptime"]

print("\nRingkasan statistik:")
print(df[fitur_numerik].describe())

# Distribusi tiap fitur
for kolom in fitur_numerik:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[kolom], kde=True, bins=30)
    plt.title(f"Distribusi {kolom}")
    plt.xlabel(kolom)
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, f"distribusi_{kolom}.png"))
    plt.close()

# Perbandingan fitur berdasarkan label
for kolom in fitur_numerik:
    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df, x="label", y=kolom)
    plt.title(f"{kolom} berdasarkan label")
    plt.xlabel("Label")
    plt.ylabel(kolom)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, f"boxplot_{kolom}.png"))
    plt.close()

# Korelasi
plt.figure(figsize=(6, 4))
corr = df[fitur_numerik].corr()
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
plt.title("Korelasi antar fitur")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "correlation_heatmap.png"))
plt.close()

# Grafik perbandingan metrik
def plot_metrics_bar(rf_m, nb_m):
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    rf_vals = [rf_m["accuracy"], rf_m["precision"], rf_m["recall"], rf_m["f1"]]
    nb_vals = [nb_m["accuracy"], nb_m["precision"], nb_m["recall"], nb_m["f1"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, rf_vals, width, label="Random Forest")
    ax.bar(x + width/2, nb_vals, width, label="Naïve Bayes")
    ax.set_ylabel("Score")
    ax.set_title("Perbandingan Model Random Forest vs Naïve Bayes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "metrics_compare.png"))
    plt.close(fig)

plot_metrics_bar(rf_metrics, nb_metrics)

# 12. Membuat kurva ROC dan PR untuk kedua model
plt.figure(figsize=(6, 4))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.3f})")
plt.plot(fpr_nb, tpr_nb, label=f"Naïve Bayes (AUC = {roc_auc_nb:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Kurva ROC – RF vs NB")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "roc_compare.png"))
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(rec_rf, prec_rf, label=f"Random Forest (AUC = {pr_auc_rf:.3f})")
plt.plot(rec_nb, prec_nb, label=f"Naïve Bayes (AUC = {pr_auc_nb:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Kurva Precision–Recall – RF vs NB")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "pr_compare.png"))
plt.close()

# 13. Menghitung dan menyimpan Feature Importance dari Random Forest
fi = rf.feature_importances_
fi_dict = dict(zip(FEATURE_NAMES, fi))

with open(os.path.join(STATIC_DIR, "feature_importance.json"), "w") as f:
    json.dump(fi_dict, f, indent=2)

plt.figure(figsize=(6, 3))
idx = np.argsort(fi)[::-1]
names_sorted = [FEATURE_NAMES[i] for i in idx]
vals_sorted = fi[idx]
plt.bar(names_sorted, vals_sorted)
plt.title("Feature Importance - Random Forest")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "feature_importance.png"))
plt.close()

# 14. Opsional: Membuat grafik SHAP jika library shap tersedia
try:
    import shap
    sample_X = X_train_s[np.random.choice(X_train_s.shape[0], min(200, X_train_s.shape[0]), replace=False)]
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(sample_X)

    if isinstance(shap_vals, list):
        shap_to_plot = shap_vals[gangguan_index]
    else:
        shap_to_plot = shap_vals

    plt.figure(figsize=(6, 4))
    shap.summary_plot(shap_to_plot, features=sample_X, feature_names=FEATURE_NAMES, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "shap_summary.png"), bbox_inches="tight")
    plt.close()
    print("SHAP summary berhasil dibuat: shap_summary.png")
except Exception as e:
    print("SHAP tidak dapat dibuat:", str(e))
    print("Jika ingin menggunakan SHAP, install dengan: pip install shap")

print("\nModel, metrik, dan visualisasi berhasil disimpan.")
print(f"Folder static: {STATIC_DIR}")
print(f"Folder models: {MODELS_DIR}")