from tensorflow.keras.models import load_model
import pandas as pd
import joblib
from scipy.stats import poisson
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from pathlib import Path

pd.set_option('display.max_columns', None)

df = pd.read_csv('1872-2026/wc.csv')
teams = pd.read_csv('1872-2026/support_data.csv')
model = load_model('models/model.keras')
preprocess = joblib.load("models/preprocess.pkl")
train = pd.read_csv('1872-2026/results.csv')

j = 0
sims = 10000
rho = 0.1
scale_home = 2.5194
scale_away = 1.8956
round_ = "1_round"

df["date"] = pd.to_datetime(df["date"])

df.sort_values(by='date',ascending=True)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek

df = df.iloc[:int(len(df)/3)]

tournament_adjust = {
"FIFA World Cup": 1,
"UEFA Euro": 1,
"Copa América": 1,
"Confederations Cup": 1,
"UEFA Nations League": 1,
"CONCACAF Nations League": 1,
"CONMEBOL–UEFA Cup of Champions": 1,
"FIFA Series": 1,
"Olympic Games": 0.8,
"FIFA World Cup qualification": 0.8,
"UEFA Euro qualification": 0.8,
"AFC Asian Cup": 0.8,
"African Cup of Nations": 0.8,
"CONCACAF Championship": 0.8,
"Gold Cup": 0.8,
"Copa América qualification": 0.8,
"AFC Asian Cup qualification": 0.8,
"African Cup of Nations qualification": 0.8,
"CONCACAF Championship qualification": 0.8,
"Oceania Nations Cup": 0.8,
"EAFF Championship": 0.8,
"AFF Championship": 0.8,
"WAFF Championship": 0.8,
"SAFF Cup": 0.8,
"COSAFA Cup": 0.8,
"UNCAF Cup": 0.8,
"CFU Caribbean Cup": 0.8,
"CCCF Championship": 0.8,
"Central American and Caribbean Games": 0.8,
"Asian Games": 0.8,
"Arab Cup": 0.8,
"NAFC Championship": 0.6,
"Pan American Championship": 0.6,
"British Home Championship": 0.6,
"Central European International Cup": 0.6,
"Nordic Championship": 0.6,
"Balkan Cup": 0.6,
"Baltic Cup": 0.6,
"Copa Roca": 0.6,
"Copa Newton": 0.6,
"Copa Lipton": 0.6,
"Copa Chevallier Boutell": 0.6,
"Copa Premio Honor Argentino": 0.6,
"Copa Premio Honor Uruguayo": 0.6,
"Copa Oswaldo Cruz": 0.6,
"Copa Río Branco": 0.6,
"Copa Bernardo O'Higgins": 0.6,
"Copa Carlos Dittborn": 0.6,
"Copa Félix Bogado": 0.6,
"Copa Juan Pinto Durán": 0.6,
"Copa del Pacífico": 0.6,
"Copa Paz del Chaco": 0.6,
"Merdeka Tournament": 0.6,
"Kirin Cup": 0.6,
"All-African Games": 0.6,
"King's Cup": 0.6,
"Korea Cup": 0.6,
"Nehru Cup": 0.6,
    "CONIFA World Football Cup": 0.6,
    "CONIFA World Football Cup qualification": 0.6,
"CONIFA European Football Cup": 0.6,
    "CONIFA South America Football Cup": 0.6,
    "CONIFA Africa Football Cup": 0.6,
    "CONIFA Asia Cup": 0.6,
"Cyprus International Tournament": 0.6,
"Four Nations Tournament": 0.6,
"Tournoi de France": 0.6,
"Rous Cup": 0.6,
"Intercontinental Cup": 0.6,
    "East Asian Games": 0.6,
    "AFF Championship qualification": 0.6,
"Mundialito": 0.6,
    "CONCACAF Series": 0.6,
"African Friendship Games": 0.6,
"Palestine Cup": 0.6,
"Friendly": 0.3,
"Gulf Cup": 0.3,
"Peace Cup": 0.3,
"Atlantic Cup": 0.3,
"Lunar New Year Cup": 0.3,
"Dunhill Cup": 0.3,
"USA Cup": 0.3,
"Merlion Cup": 0.3,
"Jordan International Tournament": 0.3,
"Simba Tournament": 0.3,
"Scania 100 Tournament": 0.3,
"Melanesia Cup": 0.3,
"South Pacific Games": 0.3,
"South Asian Games": 0.3,
    "ConIFA Challenger Cup": 0.3,
    "Southeast Asian Games": 0.3,
"Pacific Games": 0.3,
"Indian Ocean Island Games": 0.3,
"Four Nations' Cup": 0.3,
"Tri Nation Tournament": 0.3,
"Canada Shield": 0.3,
"Bolivarian Games": 0.3,
    "CONIFA World Cup qualification": 0.3,
    "West African Cup": 0.3,
"Southeast Asian Peninsular Games": 0.3,
"Phillip Seaga Cup": 0.3,
    "Oceania Nations Cup qualification": 0.3,
    "CFU Caribbean Cup qualification": 0.3,
    "COSAFA Cup qualification": 0.3,
    "Gold Cup qualification": 0.3,
"Prime Minister's Cup": 0.3,
    "MSG Prime Minister's Cup": 0.3,
    "AFC Challenge Cup": 0.3,
    "ASEAN Championship": 0.3,
"Brazil Independence Cup": 0.3,
    "Copa Confraternidad": 0.3,
    "UNIFFAC Cup": 0.3,
    "CECAFA Cup": 0.3,
    "Nations Cup": 0.3,
    "CONCACAF Nations League qualification": 0.3,
    "ASEAN Championship qualification": 0.1,
    "King Hassan II Tournament": 0.1,
"ABCS Tournament": 0.1,
    "Navruz Cup": 0.1,
    "Palestine International Championship": 0.1,
    "Philippine Peace Cup": 0.1,
    "Niamh Challenge Cup": 0.1,
    "Nile Basin Tournament": 0.1,
    "OSN Cup": 0.1,
    "Superclásico de las Américas": 0.1,
    "Outrigger Challenge Cup": 0.1,
    "Pacific Mini Games": 0.1,
    "Al Ain International Cup": 0.1,
    "Corsica Cup": 0.1,
    "Atlantic Heritage Cup": 0.1,
    "Benedikt Fontana Cup": 0.1,
    "Canadian Shield": 0.1,
    "CAFA Nations Cup": 0.1,
    "AFC Solidarity Cup": 0.1,
    "AFC Challenge Cup qualification": 0.1,
    "Coupe de l'Outre-Mer": 0.1,
    "VFF Cup": 0.1,
    "Millennium Cup": 0.1,
    "Cup of Ancient Civilizations": 0.1,
    "TIFOCO Tournament": 0.1,
    "EAFF Championship qualification": 0.1,
    "Hungary Heritage Cup": 0.1,
    "Inter Games": 0.1,
    "Morocco, Capital of African Football": 0.1,
    "Mukuru 4 Nations": 0.1,
    "Kirin Challenge Cup": 0.1,
    "South Asian Super Cup": 0.1,
"Three Nations Cup": 0.1,
    "Tri-Nations Series": 0.1,
    "Tynwald Hill Tournament": 0.1,
    "Afro-Asian Games": 0.1,
    "Mahinda Rajapaksa Cup": 0.1,
    "Mapinduzi Cup": 0.1,
    "Marianas Cup": 0.1,
    "Mauritius Four Nations Cup": 0.1,
    "Dragon Cup": 0.1,
    "SKN Football Festival": 0.1,
    "Kuneitra Cup": 0.1,
    "Trans-Tasman Cup": 0.1,
    "Great Wall Cup": 0.1,
    "UDEAC Cup": 0.1,
    "Miami Cup": 0.1,
    "Amílcar Cabral Cup": 0.1,
    "United Arab Emirates Friendship Tournament": 0.1,
    "Joe Robbie Cup": 0.1,
    "FIFA 75th Anniversary Cup": 0.1,
    "Guangzhou International Friendship Tournament": 0.1,
    "South Pacific Mini Games": 0.1,
    "Beijing International Friendship Tournament": 0.1,
    "Real Madrid 75th Anniversary Cup": 0.1,
    "Copa Ramón Castilla": 0.1,
"Indonesia Tournament": 0.1,
"Windward Islands Tournament": 0.1,
"Copa Rio Branco": 0.1,
"Vietnam Independence Cup": 0.1,
"Muratti Vase": 0.1,
    "Arab Cup qualification": 0.1,
    "Malta International Tournament": 0.1,
    "Matthews Cup": 0.1,
    "Tournament Burkina": 0.1,
    "Marlboro Cup": 0.1,
    "Island Games": 0.1,
    "NAFU Championship": 0.1,
    "Dynasty Cup": 0.1,
    "Dakar Tournament": 0.1,
"Inter-Allied Games": 0.1,
"Far Eastern Championship Games": 0.1,
"Soccer Ashes": 0.1,
"Open International Championship": 0.1,
"Zambian Independence Tournament": 0.1,
"Copa Artigas": 0.1,
"GaNEFo": 0.1,
"FIFI Wild Cup": 0.1,
    "Tournament Burkina Faso": 0.1,
"ELF Cup": 0.1,
"Viva World Cup": 0.1,
"CONIFA tournaments": 0.1,
"International Tournament of Peoples, Cultures and Tribes": 0.1,
"The Other Final": 0.1,
"Unity Cup": 0.1,
"World Unity Cup": 0.1,
}
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
    "away_dg_1": 0,
    "home_gaf_5": 0,
    "home_gaf_4": 0,
    "home_gaf_3": 0,
    "home_gaf_2": 0,
    "home_gaf_1": 0,
    "home_gog_5": 0,
    "home_gog_4": 0,
    "home_gog_3": 0,
    "home_gog_2": 0,
    "home_gog_1": 0,
    "away_gaf_5": 0,
    "away_gaf_4": 0,
    "away_gaf_3": 0,
    "away_gaf_2": 0,
    "away_gaf_1": 0,
    "away_gog_5": 0,
    "away_gog_4": 0,
    "away_gog_3": 0,
    "away_gog_2": 0,
    "away_gog_1": 0,
}

df = df.assign(**new_columns)

df["tournament_weight"] = df["tournament"].map(tournament_adjust)

df.drop(columns=["home_score", "away_score", 'city', 'tournament_weight'], inplace=True)

teams["dg"] = teams["dg"].apply(ast.literal_eval)
teams["gaf"] = teams["gaf"].apply(ast.literal_eval)
teams["gog"] = teams["gog"].apply(ast.literal_eval)

def tau(x, y, mu_home, mu_away, rho):
    """
    Corrección Dixon-Coles para marcadores bajos.
    Solo afecta: 0-0, 1-0, 0-1, 1-1
    """
    if x == 0 and y == 0:
        return 1 - mu_home * mu_away * rho
    elif x == 0 and y == 1:
        return 1 + mu_home * rho
    elif x == 1 and y == 0:
        return 1 + mu_away * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

def dixon_coles_probs(mu_home, mu_away, rho, max_goals=10):
    """
    Construye la matriz de probabilidades de marcadores
    aplicando la corrección Dixon-Coles.

    Retorna una matriz (max_goals x max_goals) con P(home=i, away=j)
    """
    probs = np.zeros((max_goals, max_goals))

    for i in range(max_goals):
        for j in range(max_goals):
            p = (poisson.pmf(i, mu_home) *
                 poisson.pmf(j, mu_away) *
                 tau(i, j, mu_home, mu_away, rho))
            probs[i, j] = p

    # Renormalizamos porque tau altera la suma total
    probs /= probs.sum()

    return probs

for i, row in df.iterrows():
    print(f'{round(j/len(df.index), 4)}')

    home_team = row["home_team"]
    away_team = row["away_team"]

    route = Path(f"predicts/{round_}/{home_team}-{away_team}")
    route.mkdir(parents=True, exist_ok=True)

    home_dg, home_gaf, home_gog, home_elo = teams.loc[teams["team"] == home_team, ["dg", "gaf", "gog", "elo"]].values[0]
    away_dg, away_gaf, away_gog, away_elo = teams.loc[teams["team"] == away_team, ["dg", "gaf", "gog", "elo"]].values[0]

    result = home_dg[-5:] + home_gaf[-5:] + home_gog[-5:] + away_dg[-5:] + away_gaf[-5:] + away_gog[-5:] + [home_elo] + [away_elo] + [home_elo - away_elo]

    df.loc[i, ["home_dg_5", "home_dg_4", "home_dg_3", "home_dg_2", "home_dg_1", "home_gaf_5", "home_gaf_4", "home_gaf_3", "home_gaf_2", "home_gaf_1", "home_gog_5", "home_gog_4", "home_gog_3", "home_gog_2", "home_gog_1", "away_dg_5", "away_dg_4", "away_dg_3", "away_dg_2", "away_dg_1", "away_gaf_5", "away_gaf_4", "away_gaf_3", "away_gaf_2", "away_gaf_1", "away_gog_5", "away_gog_4", "away_gog_3", "away_gog_2", "away_gog_1", "elo_home", "elo_away", "elo_diff"]] = result

    X = preprocess.transform(df.loc[[i]])

    resultado = model.predict(X)
    mu_home = resultado[0][0][0] * scale_home
    mu_away = resultado[1][0][0] * scale_away

    probs = dixon_coles_probs(mu_home, mu_away, rho)

    flat_probs = probs.flatten()
    indices = np.arange(len(flat_probs))
    max_goals = probs.shape[0]

    sampled = np.random.choice(indices, size=sims, p=flat_probs)

    results = [(idx // max_goals, idx % max_goals) for idx in sampled]
    results.append((mu_home, mu_away))

    match_results = pd.DataFrame(results, columns=["home_score", "away_score"])
    match_results.to_csv(f"predicts/{round_}/{home_team}-{away_team}/simulations.csv", index=False)

    pivot = pd.crosstab(match_results["home_score"][:-1], match_results["away_score"][:-1], normalize="all") * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
    plt.xlabel(away_team + " goals")
    plt.ylabel(home_team + " goals")
    plt.title(f'Predicts: {float(match_results["home_score"][len(match_results) - 1]), float(match_results["away_score"][len(match_results) - 1])}')
    plt.savefig(f"predicts/{round_}/{home_team}-{away_team}/heatmap.png")

    j += 1
