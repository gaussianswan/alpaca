from alpaca.trading.stream import TradingStream
from dotenv import load_dotenv

import requests
import os
import sys

token = '5781019360:AAHHE8_MV4qpgZE6U56o-otnhwZ2kVevNYQ'
chat_id = 1634990243


def send_telegram_message(message: str, bot_token: str = token, chat_id: int = chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={message}"

    try:
        response = requests.get(url)

        if not response:
            print(response.status_code)

    except Exception as e:
        print(e)

load_dotenv(dotenv_path=r'C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca\.env')
api_key = os.environ['ALPACA_PAPER_TRADING_KEY_ID']
secret_key = os.environ['ALPACA_PAPER_TRADING_SECRET_KEY']

trading_stream = TradingStream(api_key = api_key, secret_key=secret_key, paper=True)

async def update_handler(data):
    # trade updates will arrive in our async handler
    send_telegram_message(message="You just got a trade", bot_token=token, chat_id = chat_id)

# subscribe to trade updates and supply the handler as a parameter
trading_stream.subscribe_trade_updates(update_handler)

# start our websocket streaming
trading_stream.run()