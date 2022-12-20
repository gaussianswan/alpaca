import pandas_ta as ta
import time
import sys
import pathlib

sys.path.insert(0, pathlib.Path(r'C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca'))

from datetime import datetime
from positions import PositionSide
from telegrambot import TelegramBot
from load_api_keys import alpaca_paper_trading_keys, telegram_bot_keys

# All the Alpaca Loaders
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.models import Order
from alpaca.data.models.bars import Bar
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class CryptoLongRSIStrategy:

    def __init__(self, symbol: str, timeframe: TimeFrame, alpaca_paper_trading_keys: alpaca_paper_trading_keys, telegram_bot_keys: telegram_bot_keys, capital: float, entry_rsi_value: int = 30, exit_rsi_value: int = 70, rsi_lookback_period: int = 14) -> None:
        self.symbol = symbol
        self.capital = capital
        self.entry_rsi_value = entry_rsi_value
        self.exit_rsi_value = exit_rsi_value
        self.timeframe = timeframe
        self.rsi_lookback_period = rsi_lookback_period
        self.__api_key = alpaca_paper_trading_keys.api_key
        self.__secret_key = alpaca_paper_trading_keys.secret_key

        self.trading_client = TradingClient(self.__api_key, self.__secret_key)
        self.crypto_data_client = CryptoHistoricalDataClient(api_key = self.__api_key, secret_key = self.__secret_key)
        self.telegram_bot = TelegramBot.from_telegram_keys(telegram_keys=telegram_bot_keys)
        self.trades = []


    def get_historical_data(self, column: str = 'vwap'):
        columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
        assert column in columns, "Column passed must be from open, high, low, close, volume, trade_count, or vwap"

        data_request = CryptoBarsRequest(
            symbol_or_symbols=self.symbol,
            end = datetime.now(),
            limit = self.rsi_lookback_period + 10,
            timeframe = self.timeframe
        )

        data = self.crypto_data_client.get_crypto_bars(request_params=data_request).df

        return data[column]



    def send_sell_market_order(self, quantity: float) -> Order:
        sell_order_params = MarketOrderRequest(
            symbol = self.symbol,
            qty = quantity,
            side = OrderSide.SELL,
            time_in_force = TimeInForce.DAY
        )

        sell_order = self.trading_client.submit_order(sell_order_params)
        time.sleep(0.5)
        sell_order_update = self.trading_client.get_order_by_client_id(sell_order.id)
        return sell_order_update

    def send_buy_market_order(self, quantity: float) -> Order:
        buy_order_params = MarketOrderRequest(
            symbol = self.symbol,
            qty = quantity,
            side = OrderSide.BUY,
            time_in_force = TimeInForce.DAY
        )

        buy_order = self.trading_client.submit_order(buy_order_params)
        time.sleep(0.5)
        buy_order_update = self.trading_client.get_order_by_client_id(buy_order.id)
        return buy_order_update

    def send_long_market_order(self) -> Order:
        quantity = self.get_shares_from_capital()
        order = self.send_buy_market_order(quantity=quantity)
        return order

    def get_latest_bar(self) -> Bar:

        latest_bar_request = CryptoLatestBarRequest(
            symbol_or_symbols=  'ETH/USD'
        )

        latest_bar = self.crypto_data_client.get_crypto_latest_bar(request_params=latest_bar_request)
        return latest_bar[self.symbol]

    def get_shares_from_capital(self):
        last_bar = self.get_latest_bar()
        total_shares = self.capital / last_bar.vwap

        return total_shares

    def generate_signal(self):
        historical_data = self.get_historical_data()

        rsi = ta.rsi(close = historical_data, length = self.rsi_lookback_period).iloc[-1]

        if rsi <= self.entry_rsi_value:
            return "LONG"
        elif rsi >= self.exit_rsi_value:
            return "EXIT"
        else:
            return "NEUTRAL"

    def get_position(self) -> dict:
        """Constructs our cumulative position from a set of trades

        Returns:
            dict: Dictionary of our position with the keys "side" and "quantity". The values for side are "LONG", "SHORT", "NEUTRAL"
        """
        traded_quantity = 0

        if len(self.trades) != 0:

            # We go through each trade and do a running total
            for trade in self.trades:
                traded_quantity += trade.qty * trade.side

        if traded_quantity > 0:
            side = PositionSide.LONG
        elif traded_quantity < 0:
            side = PositionSide.SHORT
        else:
            side = PositionSide.NEUTRAL

        return {"side": side, "quantity": traded_quantity}


    def run(self):

        self.telegram_bot.send_message(message = "Running the RSI strategy")
        signal = self.generate_signal()
        current_position = self.get_position()
        total_quantity = self.get_shares_from_capital()

        if signal == 'LONG':

            # Check our positions to see if we are long already

            if current_position['side'] in [PositionSide.NEUTRAL, PositionSide.SHORT]:
                if current_position['side'] == PositionSide.NEUTRAL:
                    msg = "We are neutral and RSI level is less than 30. Going long!"
                    self.telegram_bot.send_message(message = msg)

                    market_order = self.send_long_market_order()
                    self.trades.append(market_order)

                    success_msg = f"Successfully went long! Bought {market_order.qty} {self.symbol} at an average price of {market_order.filled_avg_price} USD. The order ID is {market_order.id}"
                    self.telegram_bot.send_message(message = success_msg)

                elif current_position['side'] == PositionSide.SHORT:
                    # Closing the position down first
                    closing_out_msg = f"Closing out our short position of {current_position['quantity']} {self.symbol}"
                    self.telegram_bot.send_message(message = closing_out_msg)
                    close_out_short_order = self.send_buy_market_order(quantity = -current_position['quantity'])
                    self.trades.append(close_out_short_order)

                    # Then we go long
                    self.telegram_bot.send_message(f"Going long after closing out the short")
                    long_order = self.send_long_market_order()
                    self.trades.append(long_order)
                    success_msg = f"Successfully went long! Bought {long_order.qty} {self.symbol} at an average price of {long_order.filled_avg_price} USD. The order ID is {long_order.id}"
                    self.telegram_bot.send_message(message = success_msg)



        elif signal == "EXIT":

            if current_position['side'] == PositionSide.LONG:
                self.telegram_bot.send_message("We are triggered to sell our position. Closing out now...")
                position_close_out = self.send_sell_market_order(quantity = current_position['quantity'])
                self.trades.append(position_close_out)
                success_msg = f"Successfully closed out our position. Sold {position_close_out.qty} {self.symbol} at an average price of {position_close_out.filled_avg_price} USD. The order ID is {position_close_out.id}"
                self.telegram_bot.send_message(message = success_msg)
        else:

            self.telegram_bot.send_message("Nothing to do right now")












