import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pandas as pd
import joblib
from scipy.stats import poisson
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast

pd.set_option('display.max_columns', None)

df = pd.read_csv('1872-2026/data.csv')
teams = pd.read_csv('1872-2026/support_data.csv')
model = load_model('models/model.keras')
preprocess = joblib.load("models/preprocess.pkl")
results = pd.read_csv('1872-2026/results.csv')

j = 0
sims = 1000
rho = 0.15

df["date"] = pd.to_datetime(df["date"])

df.sort_values(by='date',ascending=True)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek

mus = results[["home_score", "away_score"]]
mus["home_mu", "away_mu"] = 0.0

teams["dg"] = teams["dg"].apply(ast.literal_eval)
teams["gaf"] = teams["gaf"].apply(ast.literal_eval)
teams["gog"] = teams["gog"].apply(ast.literal_eval)

for i, row in df.iterrows():
    print(f'{round(j/len(df.index), 4)}')

    home_team = row["home_team"]
    away_team = row["away_team"]

    X = preprocess.transform(df.loc[[i]])

    resultado = model.predict(X)
    mus.loc[i, ["home_mu", "away_mu"]] = float(resultado[0][0][0]), float(resultado[1][0][0])

    j += 1


# Sobre tu set de entrenamiento
mean_real_home = mus['home_score'].mean()   # ej: 1.35
mean_predicted_home = mus['home_mu'].mean() # ej: 0.73

mean_real_away = mus['away_score'].mean()
mean_predicted_away = mus['away_mu'].mean()

scale_home = mean_real_home / mean_predicted_home  # ≈ 1.85
scale_away = mean_real_away / mean_predicted_away

print(f'scale_home = {round(scale_home, 4)}')
print(f'scale_away = {round(scale_away, 4)}')