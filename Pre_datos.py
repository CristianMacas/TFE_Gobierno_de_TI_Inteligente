import pandas as pd

# 1. Cargar datos
ruta = "Detalles de Incidencias Cerrados.xlsx"
df = pd.read_excel(ruta, sheet_name=0)

# 2. Filtrar incidencias cerradas
df = df[df["Estado"] == "Cerrado"].copy()

# 3. Calcular tiempo de resolución en horas
df["Tiempo_Resolucion_horas"] = (
    df["Fecha_Cierre"] - df["Fecha_Creacion"]
).dt.total_seconds() / 3600

# 4. Crear variable objetivo: SLA sintético de 72 horas
SLA_LIMITE_HORAS = 72
df["sla_incumplido"] = (df["Tiempo_Resolucion_horas"] > SLA_LIMITE_HORAS).astype(int)

# 5. Variables de fecha (a partir de la fecha de creación)
df["fecha_creacion"] = pd.to_datetime(df["Fecha_Creacion"])
df["anio"] = df["fecha_creacion"].dt.year
df["mes"] = df["fecha_creacion"].dt.month
df["dia_semana"] = df["fecha_creacion"].dt.dayofweek  # 0=lunes
df["hora_creacion"] = df["fecha_creacion"].dt.hour
df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)

# 6. Seleccionar solo las columnas que usaremos en el modelo
columnas_modelo = [
    # objetivo
    "sla_incumplido",
    # tiempos
    "Tiempo_Resolucion_horas",
    # categóricas de negocio
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
    # proceso
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

df_modelo = df[columnas_modelo].copy()

# 7. (Opcional) Guardar el dataset preparado para usarlo en el notebook del modelo
df_modelo.to_csv("dataset_incidencias_preparado.csv", index=False)
print(df_modelo.head())
