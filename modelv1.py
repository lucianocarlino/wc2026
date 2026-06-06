import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

df = pd.read_csv('1872-2026/results.csv')

df = df.sort_values(by='date')

train_size = int(len(df) * 0.8)

df["date"] = pd.to_datetime(df["date"])

df = df.drop(columns='city')

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek

wcdf = df[df["home_score"].isna() | df["away_score"].isna()]
wcdf.to_csv("wc.csv", index=False)

df = df.dropna(subset=['home_score', 'away_score'])

num_cols = ["year", "month", "dayofweek"]
cat_cols = ["home_team", "away_team", "tournament", "country", "neutral"]

Xtrain = df.iloc[:train_size].drop(columns=["away_score", "home_score"])
Ytrain_away = df.iloc[:train_size].away_score
Ytrain_home = df.iloc[:train_size].home_score
Xtest = df.iloc[train_size:].drop(columns=["away_score", "home_score"])
Ytest_away = df.iloc[train_size:].away_score
Ytest_home = df.iloc[train_size:].home_score

home_model = Sequential([
    Dense(128, activation='relu', input_shape=(1038,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear'),
])

away_model = Sequential([
    Dense(128, activation='relu', input_shape=(1038,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear'),
])

home_model.compile(optimizer="adam", loss="mse", metrics=["mse"])
away_model.compile(optimizer="adam", loss="mse", metrics=["mse"])

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

scaler = StandardScaler()

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ("num", StandardScaler(), num_cols)
])

encoder_Xtrain = preprocess.fit_transform(Xtrain)
encoder_Xtest = preprocess.transform(Xtest)

home_history = home_model.fit(
    encoder_Xtrain,
    Ytrain_home,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

away_history = away_model.fit(
    encoder_Xtrain,
    Ytrain_away,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

home_loss, home_mae = home_model.evaluate(
    encoder_Xtest,
    Ytest_home
)

away_loss, away_mae = away_model.evaluate(
    encoder_Xtest,
    Ytest_away
)

print(f'home loss = {home_loss:.3f}, away loss = {away_loss:.3f}')
print(f'home mae = {home_mae:.3f}, away mae = {away_mae:.3f}')

# home loss = 2.275, away loss = 1.621
# home mae = 2.275, away mae = 1.621
