import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('1872-2026/results.csv')

df = df[df["tournament"] == "FIFA World Cup"]
df = df[["home_score", "away_score"]]
df = df.dropna(subset=['home_score', 'away_score'])

df["result"] = df["home_score"].astype(str) + df["away_score"].astype(str)

print(df['result'].value_counts(normalize=True))