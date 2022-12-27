import os
from dotenv import load_dotenv
from dataclasses import dataclass
from collections import namedtuple

@dataclass
class AlpacaPaperTradingKeys:

    api_key: str
    secret_key: str

@dataclass
class TelegramBotKeys:

    bot_token: str
    chat_id: str

# alpaca_paper_trading_keys = namedtuple(
#         "AlpacaPaperTradingKeys",
#         field_names=['api_key', 'secret_key']
#     )

# telegram_bot_keys = namedtuple(
#         "TelegramBotKeys",
#         field_names=['bot_token', 'chat_id']
#     )


def load_alpaca_paper_trading_keys(dotenv_path: str = None) -> AlpacaPaperTradingKeys:

    load_dotenv(dotenv_path=dotenv_path)

    api_key = os.environ['ALPACA_PAPER_TRADING_API_KEY']
    secret_key = os.environ['ALPACA_PAPER_TRADING_SECRET_KEY']
    keys = AlpacaPaperTradingKeys(api_key=api_key, secret_key=secret_key)

    return keys

def load_telegram_bot_keys(dotenv_path: str = None) -> TelegramBotKeys:


    load_dotenv(dotenv_path=dotenv_path)

    bot_token = os.environ['TELEGRAM_BOT_TOKEN']
    chat_id = os.environ['TELEGRAM_BOT_CHANNEL_ID']
    keys = TelegramBotKeys(bot_token=bot_token, chat_id=chat_id)
    return keys
