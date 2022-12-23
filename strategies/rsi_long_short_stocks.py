import pandas as pd
import pandas_ta as ta
import os
import pickle

from collections import namedtuple


from argparse import ArgumentParser
from math import floor
from datetime import datetime, timedelta, date, timezone
from telegrambot import telegram_bot_keys, TelegramBot
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrameUnit
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, GetCalendarRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Calendar, Position, Order, PositionSide
from alpaca.common.exceptions import APIError
from typing import List, Union

from utils import filter_trading_day, check_if_market_open, get_market_time
from load_api_keys import load_alpaca_paper_trading_keys, load_telegram_bot_keys, telegram_bot_keys, alpaca_paper_trading_keys

class StocksRSILongShortStrategy:

    def __init__(self, symbol: str, capital: float, timeframe: TimeFrame, rsi_overbought_level: int,
    rsi_oversold_level: int, stop_loss_sigma: float, take_profit_sigma: float, timeout: pd.Timedelta,
    telegram_keys: telegram_bot_keys, alpaca_paper_trading_keys: alpaca_paper_trading_keys) -> None:

        self.symbol = symbol
        self.capital = capital
        self.timeframe = timeframe
        self.rsi_overbought_level = rsi_overbought_level
        self.rsi_oversold_level = rsi_oversold_level
        self.stop_loss_sigma = stop_loss_sigma
        self.take_profit_sigma = take_profit_sigma
        self.timeout = timeout
        self.telegram_keys = telegram_keys
        self.alpaca_paper_trading_keys = alpaca_paper_trading_keys

        self.telegram_bot = TelegramBot.from_telegram_keys(telegram_keys=self.telegram_keys)
        self.trading_client = TradingClient(
            api_key=self.alpaca_paper_trading_keys.api_key,
            secret_key=self.alpaca_paper_trading_keys.secret_key
        )

        self.historical_data_client = StockHistoricalDataClient(
            api_key=self.alpaca_paper_trading_keys.api_key,
            secret_key=self.alpaca_paper_trading_keys.secret_key
        )

        self.current_position = None
        self.last_trade_time: pd.Timestamp = None
        self.market_orders: List[Order] = []


    def _get_historical_data(self) -> pd.DataFrame:
        data_request = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            start = datetime.now() - timedelta(days = 1),
            end = datetime.now(),
            timeframe = self.timeframe
        )

        data = self.historical_data_client.get_stock_bars(request_params=data_request)

        return data.df.droplevel(level = 0)

    def _get_rsi(self, column: str = 'close'):
        """Gets the most recent RSI value from the historical data

        Args:
            column (str, optional): Column used based on the OHLCV bars. Defaults to 'close'.

        Returns:
            _type_: _description_
        """

        historical_data = self._get_historical_data()
        rsi_series = ta.rsi(close = historical_data[column]).dropna()

        return rsi_series.iloc[-1]

    def _get_latest_bar(self):
        latest_bar_request_params = StockLatestBarRequest(symbol_or_symbols=self.symbol)
        latest_bar = self.historical_data_client.get_stock_latest_bar(request_params=latest_bar_request_params)

        return latest_bar[self.symbol]

    def _place_market_order(self, quantity: int, side: OrderSide, take_profit_price: float = None, stop_price: float = None, time_in_force: TimeInForce = TimeInForce.DAY, msg: str = None):
        """Sends in a market order based on a certain quantity, side, and time_in_force. You can even send a message with this

        Args:
            quantity (int): Quantity to buy or sell
            side (OrderSide): Side to take in this trade
            time_in_force (TimeInForce.DAY): Time In Force
            msg (str, optional): Optional Message to send with this trade. Defaults to None.
        """
        market_order_request = MarketOrderRequest(
            symbol = self.symbol,
            qty = quantity,
            side = side,
            time_in_force = TimeInForce.DAY
        )


        # Then we send in the order
        submitted_order = self.trading_client.submit_order(order_data = market_order_request)
        self.market_orders.append(submitted_order)
        self.telegram_bot.send_message(message=msg)

    def _close_position(self, original_side: PositionSide, quantity: int, message: str):
        if original_side == PositionSide.LONG:
            order_side = OrderSide.SELL
        elif original_side == PositionSide.SHORT:
            order_side = OrderSide.BUY

        self._place_market_order(
            quantity=quantity,
            side = order_side,
            msg = message
        )

    def _calculate_number_of_shares(self, price: float) -> int:
        return floor(self.capital / price)

    def _get_current_position(self) -> Union[Position, None]:

        try:
            current_position = self.trading_client.get_open_position(symbol_or_asset_id=self.symbol)
        except APIError:
            current_position = None

        return current_position

    def _get_trading_calendar(self) -> List[Calendar]:

        calendar_request = GetCalendarRequest(
            start = date.today()
        )

        calendar = self.trading_client.get_calendar(filters=calendar_request)

        return calendar

    def _can_trade(self) -> bool:
        """Determines if we can trade. We are going to trade when the market is open today and we are not within the last 15 minutes of the trading day

        Returns:
            bool: _description_
        """
        calendar = self._get_trading_calendar()

        if check_if_market_open(days = calendar):
            close_time = get_market_time(days = calendar, market_time='close')

            if datetime.now() < close_time - timedelta(minutes=15):
                return True
            else:
                return False
        else:
            return False

    def create_filename(self) -> str:

        string = f"RSI_long_short_{self.symbol}_rsi_high_{self.rsi_overbought_level}_rsi_low_{self.rsi_oversold_level}.pkl"

        return string

    def run(self):
        """Runs the strategy. The flow goes like this
        1. We check if we have any positions.
        2. Get the current RSI value
        3. Decide what to do (Go long, short, or wait)
        """
        # Check position side and quantity
        current_position = self._get_current_position()

        if self._can_trade():

            rsi_value = self._get_rsi()

            if current_position:
                most_recent_trade = self.market_orders[-1]
                current_side = current_position.side
                current_quantity = current_position.qty

                elapsed_time = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timestamp(most_recent_trade.filled_at)

                if elapsed_time > self.timeout:
                    # In this case we close out positions
                    self._close_position(
                        original_side=current_side,
                        quantity=current_quantity,
                        message="Closing out position since time has elapsed"
                    )

                else:

                    if current_side == PositionSide.LONG:
                        if rsi_value >= self.rsi_overbought_level:
                            self._close_position(
                                original_side=current_side,
                                quantity=current_quantity,
                                message=f"Closing out long position since the rsi value reached {rsi_value:.1f}"
                            )

                    elif current_side == PositionSide.SHORT:
                        if rsi_value <= self.rsi_oversold_level:
                            self._close_position(
                                original_side=current_side,
                                quantity=current_quantity,
                                message=f"Closing out short position since the rsi value reached {rsi_value:.1f}"
                            )

            else:
                # This is the case that we don't have any position
                latest_bar = self._get_latest_bar()
                price = latest_bar.vwap
                shares = self._calculate_number_of_shares(price = price)

                if rsi_value >= self.rsi_overbought_level:
                    self._place_market_order(
                        quantity=shares,
                        side=OrderSide.SELL,
                        msg = f'Selling {shares} shares of {self.symbol} because the RSI value is at {rsi_value:.1f}'
                    )

                elif rsi_value <= self.rsi_oversold_level:
                    self._place_market_order(
                        quantity=shares,
                        side = OrderSide.BUY,
                        msg = f"Buying {shares} shares of {self.symbol} because the RSI value is at {rsi_value:.1f}"
                    )


        else:
            # We should be closing out all positions because we are out of market hours
            if current_position:
                self._close_position(
                    original_side=current_position.side,
                    quantity=current_position.qty,
                    message="Closing out positions since the market is closed!"
                )


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument('-s', '--symbol')
    # parser.add_argument('-c', '--capital')
    # parser.add_argument('-t', '--timeframe')


    # args = parser.parse_args()

    # print(args.symbol)
    # print(args.capital)

    paper_trading_keys = load_alpaca_paper_trading_keys(dotenv_path=r'C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca\.env')
    telegram_keys = load_telegram_bot_keys(dotenv_path=r'C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca\.env')

    symbol = 'AAPL'
    capital = 1000
    timeframe = TimeFrame(amount = 2, unit = TimeFrameUnit.Minute)
    rsi_overbought_level = 70
    rsi_oversold_level = 30
    stop_loss_sigma = 0.5
    take_profit_sigma = 0.5
    timeout = pd.Timedelta(hours = 1)

    rsi_strategy = StocksRSILongShortStrategy(
        symbol=symbol,
        capital=capital,
        timeframe=timeframe,
        rsi_overbought_level=rsi_overbought_level,
        rsi_oversold_level=rsi_oversold_level,
        stop_loss_sigma=stop_loss_sigma,
        take_profit_sigma=take_profit_sigma,
        timeout = timeout,
        telegram_keys=telegram_keys,
        alpaca_paper_trading_keys=paper_trading_keys
    )

    filename = rsi_strategy.create_filename()
    filepath = f'livestrats/{filename}'

    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            rsi_strategy = pickle.load(file = f)

    # Running the strategy
    rsi_strategy.run()

    # Saving the strategy back down to our folder
    with open(filepath, 'w') as f:
        pickle.dump(rsi_strategy, f)




