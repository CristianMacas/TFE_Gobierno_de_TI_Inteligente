import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
BASE_DIR = Path(r"C:\Desarrollo Maestria\V2")

IN_CSV = BASE_DIR / "02_processed" / "dataset_incidencias_preparado_v2.csv"
OUT_CSV = BASE_DIR / "04_outputs" / "resultados_modelo_SLA_temprano_v2.csv"

MODEL_DIR = BASE_DIR / "03_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "modelo_sla_temprano_v2.joblib"
METRICS_FILE = MODEL_DIR / "metricas_modelo_v2.json"
CONFIG_FILE = MODEL_DIR / "config_modelo_v2.json"
COEF_FILE = MODEL_DIR / "coeficientes_logreg_v2.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.30

# Si tu CSV está guardado con ; y decimal , (como lo sugerimos)
CSV_SEP = ";"
CSV_DECIMAL = ","
CSV_ENCODING = "utf-8"

TARGET_COL = "sla_incumplido"


# ==========================================================
# 1) CARGA DATA
# ==========================================================
if not IN_CSV.exists():
    raise FileNotFoundError(f"No existe el archivo de entrada: {IN_CSV}")

df = pd.read_csv(IN_CSV, sep=CSV_SEP, decimal=CSV_DECIMAL, encoding=CSV_ENCODING)

if TARGET_COL not in df.columns:
    raise ValueError(f"No existe la columna objetivo '{TARGET_COL}' en el CSV.")

# Limpieza mínima de tipos numéricos clave (por si vienen como texto)
for col in ["anio", "mes", "dia_semana", "hora_creacion", "es_fin_de_semana"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# El target debe ser 0/1
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

# ==========================================================
# 2) FEATURES TEMPRANAS (solo lo disponible al crear el ticket)
# ==========================================================
features_tempranas_base = [
    "Impacto",
    "Prioridad",
    "Nivel_1_Categorizacion",
    "Nivel_2_Categorizacion",
    "Nivel_3_Categorizacion",
    "Servicio",
    "Tipo_Incidencia",
    "Fuente_Reportada",
    "anio",
    "mes",
    "dia_semana",
    "hora_creacion",
    "es_fin_de_semana",
]

# nos quedamos solo con las que existan
features_tempranas = [c for c in features_tempranas_base if c in df.columns]
if len(features_tempranas) == 0:
    raise ValueError("No se encontraron columnas de features tempranas en el dataset.")

X = df[features_tempranas].copy()
y = df[TARGET_COL].copy()

# ==========================================================
# 3) DETECTAR COLUMNAS NUMÉRICAS / CATEGÓRICAS
# ==========================================================
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

# Asegurar que columnas numéricas sean numéricas (por si alguna se coló como object)
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# ==========================================================
# 4) PIPELINE DE PREPROCESADO + MODELO (IA LIGERA)
# ==========================================================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop",
)

clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", clf),
    ]
)

# ==========================================================
# 5) TRAIN / TEST
# ==========================================================
# Si por alguna razón solo hay 1 clase, stratify fallaría; lo controlamos
n_classes = y.nunique(dropna=True)

stratify_opt = y if n_classes > 1 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=stratify_opt,
)

# ==========================================================
# 6) ENTRENAR
# ==========================================================
model.fit(X_train, y_train)

# ==========================================================
# 7) EVALUAR
# ==========================================================
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n=== Matriz de confusión (V2) ===")
print(cm)

print("\n=== Reporte de clasificación (V2) ===")
print(classification_report(y_test, y_pred, digits=3))

# ==========================================================
# 8) PROBABILIDAD PARA TODO EL DATASET + EXPORT POWER BI
# ==========================================================
proba_incumplimiento = model.predict_proba(X)[:, 1]

df_out = df.copy()
df_out["prob_incumplir_SLA_temprano"] = np.round(proba_incumplimiento.astype(float), 6)

# Guardar salida para Power BI
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(
    OUT_CSV,
    index=False,
    sep=CSV_SEP,
    decimal=CSV_DECIMAL,
    encoding=CSV_ENCODING,
    float_format="%.6f",
)

print("\nOK - CSV para Power BI generado:", OUT_CSV)

# ==========================================================
# 9) GUARDAR ARTEFACTOS EN 03_model
# ==========================================================
# 9.1 Guardar modelo (pipeline completo)
joblib.dump(model, MODEL_FILE)
print("OK - Modelo guardado:", MODEL_FILE)

# 9.2 Guardar métricas (evidencia)
metrics = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "input_csv": str(IN_CSV),
    "output_csv": str(OUT_CSV),
    "rows_total": int(len(df)),
    "rows_train": int(len(X_train)),
    "rows_test": int(len(X_test)),
    "pct_incumplimiento_sla": float(y.mean()),
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "confusion_matrix": cm.tolist(),
}

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("OK - Métricas guardadas:", METRICS_FILE)

# 9.3 Guardar configuración (para trazabilidad)
config = {
    "timestamp": metrics["timestamp"],
    "target": TARGET_COL,
    "features_tempranas_base": features_tempranas_base,
    "features_usadas": features_tempranas,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "model": "LogisticRegression",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "csv_sep": CSV_SEP,
    "csv_decimal": CSV_DECIMAL,
    "csv_encoding": CSV_ENCODING,
}

with open(CONFIG_FILE, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
print("OK - Config guardada:", CONFIG_FILE)

# 9.4 (Opcional) Guardar coeficientes (interpretabilidad)
# Esto puede fallar si scikit-learn cambia internamente; lo protegemos.
try:
    preprocess_fitted = model.named_steps["preprocess"]
    clf_fitted = model.named_steps["clf"]

    # Nombres de variables
    feature_names = []

    # numéricas
    feature_names.extend(numeric_cols)

    # categóricas one-hot
    if len(categorical_cols) > 0:
        cat_pipe = preprocess_fitted.named_transformers_["cat"]
        ohe = cat_pipe.named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_feature_names)

    coefs = clf_fitted.coef_[0]
    df_coefs = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df_coefs["abs_coef"] = df_coefs["coef"].abs()
    df_coefs = df_coefs.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])

    df_coefs.to_csv(COEF_FILE, index=False, encoding="utf-8")
    print("OK - Coeficientes guardados:", COEF_FILE)
except Exception as e:
    print("Aviso: no se pudieron guardar coeficientes (no afecta al resultado):", str(e))

print("\n=== Proceso completado V2 ===")
