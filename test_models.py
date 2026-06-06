from tensorflow.keras.models import load_model
import pandas as pd
import joblib
from scipy.stats import poisson
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

home_model = load_model("models/home_modelV2.keras")
away_model = load_model("models/away_modelV2.keras")

preprocess = joblib.load("models/preprocessV2.pkl")

match = pd.DataFrame([{
    "home_team": "Wales",
    "away_team": "Ghana",
    "tournament": "Friendly",
    "city": "Cardiff",
    "country": "Wales",
    "neutral": 0,
    "year": 2026,
    "month": 6,
    "dayofweek": 2
}])

X = preprocess.transform(match)

home_lambda = home_model.predict(X)
away_lambda = away_model.predict(X)

def simulate(home_lambda, away_lambda, sims=10000):
    results = []
    for _ in range(sims):
        home_goals = poisson.rvs(home_lambda[0][0])
        away_goals = poisson.rvs(away_lambda[0][0])
        results.append((home_goals, away_goals))

    return results

results = simulate(home_lambda, away_lambda)

df = pd.DataFrame(results, columns=["home", "away"])

pivot = pd.crosstab(df["home"], df["away"], normalize="all") * 100

plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
plt.xlabel(match["away_team"].values[0] + " goals")
plt.ylabel(match["home_team"].values[0] + " goals")
plt.title("Match score probability heatmap")
plt.show()