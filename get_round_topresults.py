import os
import pandas as pd

folder = "1_round"

results = []

for i in os.listdir(f"predicts/{folder}"):
    if not i.endswith(".csv"):
        df = pd.read_csv(f"predicts/{folder}/{i}/simulations.csv")
        conteo = (df.groupby(['home_score', 'away_score'])
               .size()
               .reset_index(name='ocurrencia')
               .sort_values('ocurrencia', ascending=False)
               .head(5))
        conteo['pct'] = round(conteo['ocurrencia'] / len(df), 4) * 100
        conteo['archivo'] = i
        results.append(conteo)

resumen_df = pd.concat(results, ignore_index=True)
resumen_df.to_excel(f"predicts/{folder}/{folder}_topresults.xlsx", index=False)