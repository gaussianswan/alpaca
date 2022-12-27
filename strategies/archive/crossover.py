
import pandas as pd
import numpy as np
import time
import requests

from datetime import datetime, timedelta

# All the Alpaca Loaders
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest, CryptoLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.models import Order
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

dotenv_path = r'C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca\.env'

class CryptoMovingAverageCrossOverStrategy:


    def __init__(self, symbol: str, short_period: int, long_period: int, alpaca_api_key: str, alpaca_secret_key: str, capital: float) -> None:

        self.capital = capital
        self.symbol = symbol
        self.short_period = short_period
        self.long_period = long_period
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.telegram_token = '5781019360:AAHHE8_MV4qpgZE6U56o-otnhwZ2kVevNYQ'
        self.telegram_chat_id = 1634990243
        self.trades = []

        self.trading_client = TradingClient(api_key=alpaca_api_key, secret_key=alpaca_secret_key)
        self.stock_data_client = StockHistoricalDataClient(api_key=alpaca_api_key, secret_key=alpaca_secret_key)
        self.crypto_data_client = CryptoHistoricalDataClient()
        self.start_date = datetime.now()
        self.end_date = None


    def get_historical_data(self):
        data_start_time = self.start_date - timedelta(days = self.long_period + 100)
        market_data_request = CryptoBarsRequest(
            symbol_or_symbols=self.symbol,
            start = data_start_time,
            timeframe=TimeFrame.Day
        )

        df = self.crypto_data_client.get_crypto_bars(request_params = market_data_request).df

        return df

    def generate_signal(self):
        """Takes the historical data and generates a buy, sell, or hold signal

        Returns:
            str: Signal which tells us what to do based on the data and current position
        """

        historical_bars = self.get_historical_data()

        # Generating the moving averages
        long_sma_column_mame = f'{self.long_period}-day SMA'
        short_sma_column_name = f'{self.short_period}-day SMA'

        historical_bars[long_sma_column_mame] = historical_bars['close'].rolling(self.long_period).mean()
        historical_bars[short_sma_column_name] = historical_bars['close'].rolling(self.short_period).mean()

        # We want to check for a crossover, only then should we go short or long
        # 1. Shorter moving average crosses above the longer one
        # 2. Shorter moving average crosses below the longer one

        prev_short_moving_average = historical_bars[short_sma_column_name].iloc[-2]
        current_short_moving_average = historical_bars[short_sma_column_name].iloc[-1]

        prev_long_moving_average = historical_bars[long_sma_column_mame].iloc[-2]
        current_long_moving_average = historical_bars[long_sma_column_mame].iloc[-1]

        if (prev_short_moving_average > prev_long_moving_average) and (current_short_moving_average < current_long_moving_average):
            ## We have a cross going down and we should go short

            signal = OrderSide.SELL

        elif (prev_short_moving_average < prev_long_moving_average) and (current_short_moving_average > current_long_moving_average):

            # We have a cross going up and we should go long
            signal = OrderSide.BUY

        else:

            signal = None

        return signal

    def calculate_shares(self):
        latest_bar_request = CryptoLatestBarRequest(symbol_or_symbols=self.symbol)

        latest_bar = self.crypto_data_client.get_crypto_latest_bar(latest_bar_request)

        vwap = latest_bar[self.symbol].vwap
        shares = self.capital / vwap
        return shares

    def send_telegram_message(self, message: str):

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage?chat_id={self.telegram_chat_id}&text={message}"

        try:
            response = requests.get(url)

            if not response:
                print(response.status_code)

        except Exception as e:
            print(e)

    def run(self):
        """Goes through the process of running the strategy and everything involved
        """
        self.send_telegram_message("Starting to run the strategy....")
        signal = self.generate_signal()
        shares_to_trade = self.calculate_shares()

        if signal == OrderSide.BUY or signal == OrderSide.SELL:

            if signal == OrderSide.BUY:
                self.send_telegram_message("The signal is LONG")
            else:
                self.send_telegram_message("The signal is SHORT")

            if len(self.positions) == 0:

                market_order_request = MarketOrderRequest(
                    symbol = self.symbol,
                    qty = shares_to_trade,
                    side = signal,
                    time_in_force = TimeInForce.DAY
                )
                self.send_telegram_message(message = "Submitting an order to create the position")
                order = self.trading_client.submit_order(market_order_request)
                time.sleep(1)
                order = self.trading_client.get_order_by_client_id(client_id=order.id)
                trade = AlpacaCryptoTrade.from_order(order = order)
                self.trades.append(trade)
                self.send_telegram_message(message = "Order submitted and position created")

            else:
                current_position = self.positions[0]
                amount_to_close = current_position.quantity

                if (current_position.side == 'Long' and signal == OrderSide.SELL) or (current_position.side == "Short" and signal == OrderSide.BUY):
                    # In this case, we have to close our long position and go short
                    self.send_telegram_message("Closing out the original order...")
                    market_order_close_out = self.create_and_submit_market_order(
                        quantity=amount_to_close,
                        side = signal
                    )

                    self.trades.append(market_order_close_out)
                    self.send_telegram_message("Setting up another order....")
                    market_order_entry = self.create_and_submit_market_order(quantity=amount_to_close, side = signal)
                    self.trades.append(market_order_entry)

        else:
            self.send_telegram_message("Nothing to do...")


    def create_and_submit_market_order(self, quantity: int, side: OrderSide) -> Order:
        """Creates and submits a market order and returns the order object for you

        Args:
            quantity (int): Quantity to trade, either shares or tokens
            side (OrderSide): Orderside which you want to sell

        Returns:
            Order: Order object
        """

        request_params = MarketOrderRequest(
            symbol = self.symbol,
            qty = quantity,
            side = side,
            time_in_force = TimeInForce.DAY
        )

        order = self.trading_client.submit_order(order_data = request_params)
        time.sleep(1)
        order = self.trading_client.get_order_by_client_id(client_id = order.id)

        return order

class AlpacaCryptoPosition:

    def __init__(self, symbol: str, qty: float, side: int) -> None:
        assert side in [-1, 1], "Side must be either 1 for long or -1 for short"
        self.symbol = symbol
        self.qty = qty
        self.side = side

class AlpacaCryptoTrade:

    def __init__(self, symbol: str, qty: float, side: OrderSide, filled_price: float, filled_at: datetime) -> None:
        self.symbol = symbol
        self.qty = qty
        self.filled_price = filled_price
        self.filled_at = filled_at
        self.side = side

    @classmethod
    def from_order(cls, order: Order):
        return cls(
            order.symbol,
            order.qty,
            order.side,
            order.filled_avg_price,
            order.filled_at
        )






