import requests

class TelegramBot:

    def __init__(self, bot_token: str, chat_id: int) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, message: str):
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={message}"

        try:
            response = requests.get(url)

            if not response:
                print(response.status_code)

        except Exception as e:
            print(e)


