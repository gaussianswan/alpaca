import os
from dotenv import load_dotenv

from collections import namedtuple

def load_alpaca_paper_trading_keys(dotenv_path: None) -> namedtuple:

    alpaca_paper_trading_keys = namedtuple(
        "AlpacaPaperTradingKeys",
        field_names=['api_key', 'secret_key']
    )

    load_dotenv(dotenv_path=dotenv_path)

    api_key = os.environ['ALPACA_PAPER_TRADING_API_KEY']
    secret_key = os.environ['ALPACA_PAPER_TRADING_SECRET_KEY']
    keys = alpaca_paper_trading_keys(api_key, secret_key)

    return keys