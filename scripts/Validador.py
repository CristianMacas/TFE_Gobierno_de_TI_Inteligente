import pandas as pd

df = pd.read_csv(r"C:\Desarrollo Maestria\V2\04_outputs\resultados_modelo_SLA_temprano_v2.csv",
                 sep=";", decimal=",", encoding="utf-8")

print("Filas:", len(df))
print("Años:", sorted(df["anio"].unique()))
print("Meses por año:")
print(df.groupby("anio")["mes"].unique())

print("\n% Incumplimiento SLA:", round(df["sla_incumplido"].mean()*100, 2))
print("Tiempo resolución (h) min/avg/max:",
      df["Tiempo_Resolucion_horas"].min(),
      df["Tiempo_Resolucion_horas"].mean(),
      df["Tiempo_Resolucion_horas"].max())

print("\nRiesgo prob (min/avg/max):",
      df["prob_incumplir_SLA_temprano"].min(),
      df["prob_incumplir_SLA_temprano"].mean(),
      df["prob_incumplir_SLA_temprano"].max())
