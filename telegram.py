import requests
import os

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