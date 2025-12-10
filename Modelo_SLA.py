# Modelo_SLA.py - versión alerta temprana

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# 1. Cargar el dataset preparado
# (dataset_incidencias_preparado.csv sigue con separador coma y punto decimal)
df = pd.read_csv("dataset_incidencias_preparado.csv")

# 2. Variable objetivo
y = df["sla_incumplido"]

# 3. Definir columnas que SÍ se conocen al crear la incidencia
features_tempranas = [
    "Impacto",
    "Prioridad",
    "Nivel_1_Categorizacion",
    "Nivel_2_Categorizacion",
    "Nivel_3_Categorizacion",
    "Servicio",
    "Tipo_Incidencia",
    "Fuente_Reportada",
    # variables temporales derivadas de Fecha_Creacion
    "anio",
    "mes",
    "dia_semana",
    "hora_creacion",
    "es_fin_de_semana",
]

X = df[features_tempranas].copy()

# 4. Detectar columnas numéricas y categóricas
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Columnas numéricas:", numeric_cols)
print("Columnas categóricas:", categorical_cols)

# 5. Transformadores
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
    ]
)

# 6. Modelo de IA ligera (Regresión Logística)
model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)

# 7. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 8. Entrenar
model.fit(X_train, y_train)

# 9. Evaluar
y_pred = model.predict(X_test)

print("\n=== Matriz de confusión (modelo alerta temprana) ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Reporte de clasificación (modelo alerta temprana) ===")
print(classification_report(y_test, y_pred, digits=3))

# 10. Probabilidad de incumplimiento para TODO el dataset
prob_incumplimiento = model.predict_proba(X)[:, 1]
df_salida = df.copy()
df_salida["prob_incumplir_SLA_temprano"] = prob_incumplimiento

# 11. Guardar resultados para Power BI
#    - sep=";"      -> separador de columnas: punto y coma
#    - decimal=","  -> separador decimal: coma
#    - float_format -> evita notación científica y fija 6 decimales
df_salida.to_csv(
    "resultados_modelo_SLA_temprano.csv",
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8",
    float_format="%.6f",
)

print("\nArchivo 'resultados_modelo_SLA_temprano.csv' generado correctamente con separador ';' y coma decimal.")
