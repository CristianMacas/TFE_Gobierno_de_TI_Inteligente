import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"C:\Desarrollo Maestria\V2")
INPUT_XLSX = BASE_DIR / "01_raw" / "Casos_Cerrados_2025_v2.xlsx"
SHEET_NAME = "1-OPC Detalles de Incidencias C"  # clave en V2

OUT_CSV = BASE_DIR / "02_processed" / "dataset_incidencias_preparado_v2.csv"

SLA_LIMITE_HORAS = 72  # Mantener para comparar todos los meses (consistente)

# =========================
# 1) Cargar datos
# =========================
df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

# =========================
# 2) Filtrar incidencias cerradas
# =========================
df = df[df["Estado"] == "Cerrado"].copy()

# =========================
# 3) Calcular tiempo de resolución (horas)
# =========================
df["Tiempo_Resolucion_horas"] = (
    df["Fecha_Cierre"] - df["Fecha_Creacion"]
).dt.total_seconds() / 3600

# Regla de calidad (opcional): quitar tiempos negativos o absurdos
df = df[df["Tiempo_Resolucion_horas"] >= 0].copy()

# =========================
# 4) Variable objetivo SLA (sintética 72h)
# =========================
df["sla_incumplido"] = (df["Tiempo_Resolucion_horas"] > SLA_LIMITE_HORAS).astype(int)

# =========================
# 5) Variables temporales (a partir de Fecha_Creacion)
# =========================
df["fecha_creacion"] = pd.to_datetime(df["Fecha_Creacion"])
df["anio"] = df["fecha_creacion"].dt.year
df["mes"] = df["fecha_creacion"].dt.month
df["dia_semana"] = df["fecha_creacion"].dt.dayofweek  # 0=lunes
df["hora_creacion"] = df["fecha_creacion"].dt.hour
df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)

# =========================
# 6) Columnas para el modelo / análisis
#    (solo usa las que existen en V2)
# =========================
columnas_modelo = [
    # objetivo
    "sla_incumplido",
    # tiempos
    "Tiempo_Resolucion_horas",
    # negocio/categorización
    "Impacto",
    "Prioridad",
    "Nivel_1_Categorizacion",
    "Nivel_2_Categorizacion",
    "Nivel_3_Categorizacion",
    "Servicio",
    "Tipo_Incidencia",
    "Fuente_Reportada",
    "Tipo_Solucion",
    "Categoria_Resolucion",
    # proceso (ojo: estas NO deben entrar al modelo "temprano", pero sí son útiles en dataset)
    "Total_Transferencias",
    "Indisponibilidad_Minutos",
    "Reapertura",
    "Incidente_Mayor",
    # fecha derivada
    "anio",
    "mes",
    "dia_semana",
    "hora_creacion",
    "es_fin_de_semana",
]

# Verificación: solo quedarnos con columnas que realmente existan
faltantes = [c for c in columnas_modelo if c not in df.columns]
if faltantes:
    print("⚠️ Columnas faltantes en el Excel (se omiten):", faltantes)
columnas_modelo = [c for c in columnas_modelo if c in df.columns]

df_modelo = df[columnas_modelo].copy()

# =========================
# 7) Guardar dataset preparado (Power BI friendly)
# =========================
df_modelo.to_csv(
    OUT_CSV,
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8",
)

print("OK - Generado:", OUT_CSV)
print(df_modelo.head(3))
print("Registros:", len(df_modelo))
print("Incumplen SLA (%):", round(df_modelo["sla_incumplido"].mean() * 100, 2))
