from operator import index
import numpy as np
import pandas as pd

df = pd.read_csv('1872-2026/results.csv')

df = df.sort_values(by='date')

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek

df["home_dg"] = None
df["away_dg"] = None

df = df[df["year"] >= 1910]

df.drop(columns=["tournament", "city", "country"], inplace=True)

home = df["home_team"].unique()
away = df["away_team"].unique()

teams = home.tolist() + away.tolist()
teams = [([0.0, 0.0, 0.0, 0.0, 0.0], i) for i in teams]
teams = pd.DataFrame(teams, columns=["dg", "team"])
teams = teams.drop_duplicates(subset=["team"])

for i in range(len(df)):
    home_dg = df.iloc()[i]["home_score"] - df.iloc()[i]["away_score"]
    away_dg = df.iloc()[i]["away_score"] - df.iloc()[i]["home_score"]
    home_team = df.iloc()[i]["home_team"]
    away_team = df.iloc()[i]["away_team"]

    print(f'home team: {home_team}, away team: {away_team}')
    teams[teams["team"] == home_team]["dg"][teams[teams["team"] == home_team].last_valid_index()].append(float(home_dg))
    teams[teams["team"] == away_team]["dg"][teams[teams["team"] == away_team].last_valid_index()].append(float(away_dg))
    print(f' fila de home team: {teams[teams["team"] == home_team]}')
    print(f'fila de away team: {teams[teams["team"] == away_team]}')
    print(f'a insertar en home: {teams[teams["team"] == home_team]["dg"][teams[teams["team"] == home_team].last_valid_index()][-5:]}')
    print(f'a insertar en away: {teams[teams["team"] == away_team]["dg"][teams[teams["team"] == away_team].last_valid_index()][-5:]}')
    print(f'home_team: {df.ix[i, home_team]}')
    input("enter")
    df.loc[i, "home_dg"] = teams[teams["team"] == home_team]["dg"][teams[teams["team"] == home_team].last_valid_index()][-5:]
    df.loc[i, "away_dg"] = teams[teams["team"] == away_team]["dg"][teams[teams["team"] == away_team].last_valid_index()][-5:]

df.to_csv("1872-2026/results.csv", index=False)