import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.optimizers import SGD
from sklearn.compose import ColumnTransformer
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv('1872-2026/data.csv')

# df = df.sort_values(by='date')

train_end = int(len(df) * 0.7)
val_end = int(len(df) * 0.8)

df = df.drop(columns=['city', 'tournament_weight'])

df = df.dropna(subset=['home_score', 'away_score'])

num_cols = ["year", "month", "dayofweek", "home_dg_5", "home_dg_4", "home_dg_3", "home_dg_2", "home_dg_1", "away_dg_5", "away_dg_4", "away_dg_3", "away_dg_2", "away_dg_1", "elo_home", "elo_away", "elo_diff", "home_gaf_5", "home_gaf_4", "home_gaf_3","home_gaf_2","home_gaf_1","home_gog_5","home_gog_4","home_gog_3","home_gog_2","home_gog_1","away_gaf_5","away_gaf_4","away_gaf_3","away_gaf_2","away_gaf_1","away_gog_5","away_gog_4","away_gog_3","away_gog_2","away_gog_1",]
cat_cols = ["home_team", "away_team", "tournament", "country", "neutral"]

Xtrain = df.iloc[:train_end].drop(columns=["away_score", "home_score"])
Ytrain_away = np.log1p(df.iloc[:train_end].away_score)
Ytrain_home = np.log1p(df.iloc[:train_end].home_score)

Xval = df.iloc[train_end:val_end].drop(columns=["away_score", "home_score"])
Yval_away = np.log1p(df.iloc[train_end:val_end].away_score)
Yval_home = np.log1p(df.iloc[train_end:val_end].home_score)

Xtest = df.iloc[val_end:].drop(columns=["away_score", "home_score"])
Ytest_away = np.log1p(df.iloc[val_end:].away_score)
Ytest_home = np.log1p(df.iloc[val_end:].home_score)


inputs = Input(shape=(1004,))

x = Dense(256, activation="relu")(inputs)
x = Dropout(0.3)(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)

shared = Dense(32, activation="relu")(x)

home_branch = Dense(16, activation="relu")(shared)
home_output = Dense(1, activation="softplus", name="home_score")(home_branch)

away_branch = Dense(16, activation="relu")(shared)
away_output = Dense(1, activation="softplus", name="away_score")(away_branch)

model = Model(
    inputs=inputs,
    outputs=[home_output, away_output]
)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs"
)

scaler = StandardScaler()

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ("num", StandardScaler(), num_cols)
])

encoder_Xtrain = preprocess.fit_transform(Xtrain)
encoder_Xval = preprocess.transform(Xval)
encoder_Xtest = preprocess.transform(Xtest)

model.compile(optimizer=SGD(learning_rate=0.01), loss={
    "home_score": "mse",
    "away_score": "mse",
}, metrics={
    "home_score": "mae",
    "away_score": "mae",
})

history = model.fit(
    encoder_Xtrain,
    {
        "home_score": Ytrain_home,
        "away_score": Ytrain_away,
    },
    validation_data=(encoder_Xval, {
        "home_score": Yval_home,
        "away_score": Yval_away,
    }),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop, tensorboard_callback],
)

results = model.evaluate(
    encoder_Xtest,
    {
        "home_score": Ytest_home,
        "away_score": Ytest_away
    },
    verbose=0
)
loss, home_loss, away_loss, home_mae, away_mae = results

history.history.keys()

print(f'home loss = {home_loss:.3f}, away loss = {away_loss:.3f}')
print(f'home mae = {home_mae:.3f}, away mae = {away_mae:.3f}')

plt.figure(figsize=(10, 5))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Total Loss')
plt.legend()
plt.grid(True)

plt.show()

nombre = input("Press ENTER to save")

model.save(f"models/model{nombre}.keras")

joblib.dump(preprocess, f"models/preprocess{nombre}.pkl")
