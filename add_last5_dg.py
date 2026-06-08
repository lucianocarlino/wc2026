import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

j = 0

df = pd.read_csv('1872-2026/results.csv')

df["date"] = pd.to_datetime(df["date"])

df.sort_values(by='date',ascending=True)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek

df = df[df["year"] >= 1910]

df = df.dropna(subset=['home_score', 'away_score'])

new_columns = {
    "home_dg_5": 0,
    "home_dg_4": 0,
    "home_dg_3": 0,
    "home_dg_2": 0,
    "home_dg_1": 0,
    "away_dg_5": 0,
    "away_dg_4": 0,
    "away_dg_3": 0,
    "away_dg_2": 0,
    "away_dg_1": 0
}

df = df.assign(**new_columns)

home = df["home_team"].unique()
away = df["away_team"].unique()

teams = home.tolist() + away.tolist()
teams = [(i, [0, 0, 0, 0, 0]) for i in teams]
teams = pd.DataFrame(teams, columns=["team", "dg"])
teams = teams.drop_duplicates(subset=["team"])

for i, row in df.iterrows():
    print(f'{j} of {len(df.index)}')
    home_match_dg = row["home_score"] - row["away_score"]
    away_match_dg = row["away_score"] - row["home_score"]
    home_team = row["home_team"]
    away_team = row["away_team"]

    teams.loc[teams["team"] == home_team, ["dg"]].values[0][0].append(int(home_match_dg))
    teams.loc[teams["team"] == away_team, ["dg"]].values[0][0].append(int(away_match_dg))

    df.loc[i, ["home_dg_5", "home_dg_4", "home_dg_3", "home_dg_2", "home_dg_1"]] = teams.loc[teams["team"] == home_team, ["dg"]].values[0][0][-6:-1]
    df.loc[i, ["away_dg_5", "away_dg_4", "away_dg_3", "away_dg_2", "away_dg_1"]] = teams.loc[teams["team"] == away_team, ["dg"]].values[0][0][-6:-1]
    j += 1

print(df.iloc[len(df.index)-1])
df.to_csv("1872-2026/resultsv2.csv", index=False)