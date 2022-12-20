import requests
from load_api_keys import telegram_bot_keys
class TelegramBot:

    def __init__(self, bot_token: str, chat_id: int) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, message: str):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage?chat_id={self.chat_id}&text={message}"

        try:
            response = requests.get(url)

            if not response:
                print(response.status_code)

        except Exception as e:
            print(e)

    @classmethod
    def from_telegram_keys(cls, telegram_keys: telegram_bot_keys):
        bot_token = telegram_keys.bot_token
        chat_id = telegram_keys.chat_id

        return cls(bot_token, chat_id)





