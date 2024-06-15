import asyncio
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
from datetime import datetime
from bitget_perp import PerpBitget

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemin des fichiers
DATA_FILE = "trade_data.csv"
MODEL_FILE = "trading_model.pkl"

# Fonction pour enregistrer les données de trading
def save_trade_data(trade_data, filename=DATA_FILE):
    if os.path.exists(filename):
        existing_data = pd.read_csv(filename)
        updated_data = pd.concat([existing_data, trade_data])
    else:
        updated_data = trade_data

    updated_data.to_csv(filename, index=False)

# Fonction pour calculer la volatilité
def calculate_volatility(prices, window=20):
    log_returns = np.log(prices / prices.shift(1))
    volatility = log_returns.rolling(window=window).std()
    return volatility

# Fonction pour ajuster la taille des positions
def adjust_position_size(base_size, volatility, target_volatility=0.02):
    if volatility > 0:
        return base_size * (target_volatility / volatility)
    else:
        return base_size

# Fonction pour placer des ordres avec gestion des risques
async def place_dynamic_order_with_risk_management(exchange, pair, side, price, usdt_balance, base_size, volatility, stop_loss_pct=0.02, take_profit_pct=0.04):
    size = adjust_position_size(base_size, volatility)
    size = exchange.amount_to_precision(pair, size / price)

    # Place main order
    main_order = await exchange.place_trigger_order(
        pair=pair,
        side=side,
        price=exchange.price_to_precision(pair, price),
        trigger_price=exchange.price_to_precision(pair, price * (1.005 if side == "buy" else 0.995)),
        size=size,
        type="limit",
        reduce=False,
        margin_mode="cross"
    )

    # Calculate stop-loss and take-profit prices
    stop_loss_price = price * (1 - stop_loss_pct) if side == "buy" else price * (1 + stop_loss_pct)
    take_profit_price = price * (1 + take_profit_pct) if side == "buy" else price * (1 - take_profit_pct)

    # Place stop-loss order
    await exchange.place_trigger_order(
        pair=pair,
        side="sell" if side == "buy" else "buy",
        price=exchange.price_to_precision(pair, stop_loss_price),
        trigger_price=exchange.price_to_precision(pair, stop_loss_price),
        size=size,
        type="limit",
        reduce=True,
        margin_mode="cross"
    )

    # Place take-profit order
    await exchange.place_trigger_order(
        pair=pair,
        side="sell" if side == "buy" else "buy",
        price=exchange.price_to_precision(pair, take_profit_price),
        trigger_price=exchange.price_to_precision(pair, take_profit_price),
        size=size,
        type="limit",
        reduce=True,
        margin_mode="cross"
    )

# Fonction pour charger et former le modèle
def train_predictive_model(filename=DATA_FILE):
    # Charger les données enregistrées
    data = pd.read_csv(filename)

    # Calculer les caractéristiques
    data['log_return'] = np.log(data['price'] / data['price'].shift(1))
    data['volatility'] = data['log_return'].rolling(window=20).std()

    # Remplacer les valeurs manquantes
    data = data.fillna(0)

    # Sélectionner les caractéristiques et la cible
    features = ['price', 'volume', 'volatility']
    target = 'side'  # Par exemple, 1 pour 'buy', 0 pour 'sell'

    X = data[features]
    y = data[target]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Former le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Évaluer le modèle
    accuracy = model.score(X_test, y_test)
    logger.info(f"Accuracy: {accuracy}")

    # Sauvegarder le modèle
    joblib.dump(model, MODEL_FILE)
    return model

# Fonction pour prédire l'action de trading
def predict_trade_action(model, price, volume, volatility):
    new_data = np.array([[price, volume, volatility]])
    predicted_side = model.predict(new_data)
    return predicted_side

# Fonction principale de trading
async def main_trading_logic(exchange, params):
    # Vérifier si le modèle existe, sinon le former
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        model = train_predictive_model()

    while True:
        # Fetch market data
        market_data = await fetch_market_data(exchange, params)

        for pair in params:
            current_price = market_data[pair]["close"][-1]
            current_volume = market_data[pair]["volume"][-1]

            # Calculer la volatilité
            volatility = calculate_volatility(market_data[pair]["close"])[-1]

            # Prédiction avec le modèle de machine learning
            predicted_side = predict_trade_action(model, current_price, current_volume, volatility)

            usdt_balance = await exchange.get_balance()

            # Place order based on the prediction
            if predicted_side == 1:
                await place_dynamic_order_with_risk_management(
                    exchange=exchange,
                    pair=pair,
                    side="buy",
                    price=current_price,
                    usdt_balance=usdt_balance.free,
                    base_size=params[pair]["size"] * usdt_balance.free / len(params[pair]["envelopes"]),
                    volatility=volatility
                )
            else:
                await place_dynamic_order_with_risk_management(
                    exchange=exchange,
                    pair=pair,
                    side="sell",
                    price=current_price,
                    usdt_balance=usdt_balance.free,
                    base_size=params[pair]["size"] * usdt_balance.free / len(params[pair]["envelopes"]),
                    volatility=volatility
                )

            # Enregistrer les données de marché
            trade_data = pd.DataFrame({
                'timestamp': [datetime.now()],
                'price': [current_price],
                'volume': [current_volume],
                'side': [predicted_side]
            })
            save_trade_data(trade_data)

        # Sleep for a while before checking the market again
        await asyncio.sleep(60)

# Fonction pour récupérer les données de marché depuis BitGet
async def fetch_market_data(exchange, params):
    market_data = {}
    for pair in params:
        df = await exchange.get_last_ohlcv(pair, timeframe='1m', limit=100)
        market_data[pair] = df
    return market_data

# Exemple de paramètres de trading
params = {
    'BTC/USDT': {
        'size': 0.1,
        'envelopes': [0.01, 0.02, 0.03],
        'canceled_orders_sell': 0
    },
    # Ajouter d'autres paires de trading ici
}

# Point d'entrée principal
if __name__ == "__main__":
    exchange = PerpBitget()
    asyncio.run(main_trading_logic(exchange, params))
