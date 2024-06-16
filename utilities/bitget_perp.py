import ccxt.async_support as ccxt
import pandas as pd

class PerpBitget:
    def __init__(self, public_api, secret_api, password):
        bitget_auth_object = {
            "apiKey": public_api,
            "secret": secret_api,
            "password": password,
            "enableRateLimit": True,
            "rateLimit": 100,
            "options": {"defaultType": "future"},
        }
        if not bitget_auth_object["secret"]:
            self._auth = False
            self._session = ccxt.bitget()
        else:
            self._auth = True
            self._session = ccxt.bitget(bitget_auth_object)
    
    async def load_markets(self):
        self.market = await self._session.load_markets()
    
    async def close(self):
        await self._session.close()
    
    def ext_pair_to_pair(self, ext_pair) -> str:
        return f"{ext_pair}:USDT"
    
    def pair_to_ext_pair(self, pair) -> str:
        return pair.replace(":USDT", "")
    
    def get_pair_info(self, ext_pair) -> str:
        return self.market.get(ext_pair, {})
    
    async def place_order(self, pair, side, amount, price=None, type='market', reduce=False, margin_mode='isolated', error=True):
        try:
            order = await self._session.create_order(
                symbol=pair, type=type, side=side, amount=amount, price=price, params={"reduce_only": reduce, "margin_mode": margin_mode}
            )
            return order
        except ccxt.BaseError as e:
            if error:
                raise e
            print(f"Error placing order: {e}")
            return None

    async def place_trigger_order(self, pair, side, trigger_price, price, amount, type='limit', reduce=False, margin_mode='isolated', error=True):
        try:
            order = await self._session.create_order(
                symbol=pair, type=type, side=side, amount=amount, price=price, params={"stopPrice": trigger_price, "reduce_only": reduce, "margin_mode": margin_mode}
            )
            return order
        except ccxt.BaseError as e:
            if error:
                raise e
            print(f"Error placing trigger order: {e}")
            return None

    async def get_balance(self):
        try:
            balance = await self._session.fetch_balance()
            return balance
        except ccxt.BaseError as e:
            print(f"Error fetching balance: {e}")
            return None

    async def get_open_positions(self, pairs):
        try:
            positions = await self._session.fetch_positions(params={"symbol": pairs})
            return positions
        except ccxt.BaseError as e:
            print(f"Error fetching positions: {e}")
            return None

    async def get_open_trigger_orders(self, pair):
        try:
            orders = await self._session.fetch_open_orders(symbol=pair, params={"type": "stop"})
            return orders
        except ccxt.BaseError as e:
            print(f"Error fetching open trigger orders: {e}")
            return None

    async def get_open_orders(self, pair):
        try:
            orders = await self._session.fetch_open_orders(symbol=pair)
            return orders
        except ccxt.BaseError as e:
            print(f"Error fetching open orders: {e}")
            return None

    async def cancel_trigger_orders(self, pair, order_ids):
        try:
            for order_id in order_ids:
                await self._session.cancel_order(id=order_id, symbol=pair)
        except ccxt.BaseError as e:
            print(f"Error canceling trigger orders: {e}")

    async def cancel_orders(self, pair, order_ids):
        try:
            for order_id in order_ids:
                await self._session.cancel_order(id=order_id, symbol=pair)
        except ccxt.BaseError as e:
            print(f"Error canceling orders: {e}")

    async def set_margin_mode_and_leverage(self, pair, margin_mode, leverage):
        try:
            await self._session.set_margin_mode(margin_mode, pair)
            await self._session.set_leverage(leverage, pair)
        except ccxt.BaseError as e:
            print(f"Error setting margin mode and leverage: {e}")

    async def get_last_ohlcv(self, pair, timeframe, limit):
        try:
            ohlcv = await self._session.fetch_ohlcv(pair, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except ccxt.BaseError as e:
            print(f"Error fetching OHLCV data: {e}")
            return None
