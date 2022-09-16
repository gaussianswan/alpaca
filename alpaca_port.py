import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, GetOrderByIdRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from dotenv import load_dotenv

import os
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, GetOrderByIdRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from dotenv import load_dotenv

class AlpacaTradingAccount:

    def __init__(self, env_path: str = None, paper = True):
        self.env_path = env_path
        self.paper = paper
        self.api_key, self.secret_key = self._load_environment()
        self.client = TradingClient(api_key=self.api_key, secret_key=self.secret_key, paper = self.paper)

    def _load_environment(self):
        if self.env_path is not None:
            load_dotenv(self.env_path)

        else:
            load_dotenv()

        # Then we are going to get the API keys
        if self.paper:
            api_key = os.environ['ALPACA_PAPER_TRADING_KEY_ID']
            secret_key = os.environ['ALPACA_PAPER_TRADING_SECRET_KEY']
        else:
            api_key = os.environ['ALPACA_LIVE_TRADING_KEY_ID']
            secret_key = os.environ['ALPACA_LIVE_TRADING_SECRET_KEY']

        return api_key, secret_key

    def get_all_active_positions(self) -> pd.DataFrame:
        positions = self.client.get_all_positions()
        positions_list = []
        for position in positions:
            position_dict = dict(position)
            positions_list.append(dict(position))

        positions_df = pd.DataFrame(positions_list)
        as_numeric_columns = ['avg_entry_price', 'qty', 'market_value', 'cost_basis', 'unrealized_pl', 'unrealized_plpc', 'unrealized_intraday_pl', 'unrealized_intraday_plpc', 'current_price', 'lastday_price', 'change_today']
        positions_df[as_numeric_columns] = positions_df[as_numeric_columns].astype(float).round(3)

        return positions_df

    def get_all_closed_positions(self):
        pass

    def get_all_active_orders(self):
        pass