import datetime
import sys
import asyncio
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import ta
from utilities.bitget_perp import PerpBitget
from secret import ACCOUNTS

sys.path.append("./Live-Tools-V2")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Chemin pour sauvegarder les données historiques et le modèle
historical_data_path = "./historical_data.pkl"
model_path = "./model.pkl"

def save_historical_data(data):
    with open(historical_data_path, 'wb') as f:
        pd.to_pickle(data, f)

def load_historical_data():
    if os.path.exists(historical_data_path):
        with open(historical_data_path, 'rb') as f:
            return pd.read_pickle(f)
    else:
        # Créer un fichier vide s'il n'existe pas
        empty_data = {}
        save_historical_data(empty_data)
        return empty_data

def combine_data(old_data, new_data):
    combined_data = {}
    for pair in new_data:
        if pair in old_data:
            combined_data[pair] = pd.concat([old_data[pair], new_data[pair]]).drop_duplicates().reset_index(drop=True)
        else:
            combined_data[pair] = new_data[pair]
    return combined_data

def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return RandomForestClassifier()

def save_model(model):
    joblib.dump(model, model_path)

def train_model(data, labels):
    model = load_model()
    model.fit(data, labels)
    save_model(model)

def predict(data):
    model = load_model()
    return model.predict(data)

async def fetch_historical_data(perp_bitget, pair, timeframe, limit=100):
    try:
        ohlcv = await perp_bitget._session.fetch_ohlcv(pair, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching historical data for {pair}: {e}")
        return None

def apply_envelope_strategy(df, window, envelope_percent):
    df['ma'] = df['close'].rolling(window=window).mean()
    df['upper_envelope'] = df['ma'] * (1 + envelope_percent)
    df['lower_envelope'] = df['ma'] * (1 - envelope_percent)
    df.dropna(inplace=True)
    return df

async def main():
    account = ACCOUNTS["bitget1"]

    margin_mode = "isolated"
    exchange_leverage = 3

    tf = "1h"
    size_leverage = 3
    sl = 0.3
    params = {
        "BTC/USDT": {"src": "close", "ma_base_window": 7, "envelopes": [0.07, 0.1, 0.15], "size": 0.1, "sides": ["long", "short"]},
        "ETH/USDT": {"src": "close", "ma_base_window": 5, "envelopes": [0.07, 0.1, 0.15], "size": 0.1, "sides": ["long", "short"]},
    }

    perp_bitget = PerpBitget(account["public_api"], account["secret_api"], account["password"])
    await perp_bitget.load_markets()

    # Charger les données historiques
    historical_data = load_historical_data()

    # Collecter les nouvelles données de marché
    new_data = {}
    for pair, settings in params.items():
        df = await fetch_historical_data(perp_bitget, pair, tf)
        if df is not None:
            new_data[pair] = df
        else:
            new_data[pair] = pd.DataFrame()

    # Combiner les nouvelles données avec les données historiques
    combined_data = combine_data(historical_data, new_data)

    # Sauvegarder les données combinées
    save_historical_data(combined_data)

    # Collecte des données et étiquettes pour l'entraînement du modèle
    data, labels = [], []
    for pair, settings in params.items():
        df = combined_data[pair]
        for envelope_percent in settings["envelopes"]:
            df = apply_envelope_strategy(df, settings["ma_base_window"], envelope_percent)
            for i in range(len(df) - 1):
                features = df.iloc[i][['close', 'ma', 'upper_envelope', 'lower_envelope']].values
                label = 1 if df.iloc[i + 1]['close'] > df.iloc[i]['close'] else 0
                data.append(features)
                labels.append(label)

    # Entraînement du modèle
    train_model(data, labels)

    # Prédictions et passage des ordres
    for pair, settings in params.items():
        df = combined_data[pair]
        for envelope_percent in settings["envelopes"]:
            df = apply_envelope_strategy(df, settings["ma_base_window"], envelope_percent)
            if not df.empty:
                features = df.iloc[-1][['close', 'ma', 'upper_envelope', 'lower_envelope']].values
                prediction = predict([features])[0]

                if prediction == 1:  # Acheter
                    await perp_bitget.place_order(pair, 'buy', settings["size"])
                else:  # Vendre
                    await perp_bitget.place_order(pair, 'sell', settings["size"])

if __name__ == "__main__":
    asyncio.run(main())
