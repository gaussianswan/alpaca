from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from load_api_keys import load_alpaca_paper_trading_keys
import datetime


class AlpacaCryptoTradingClient:

    def __init__(self, api_key: str, secret_key: str) -> None:

        self.client = TradingClient(api_key=api_key, secret_key=secret_key)

    def get_all_crypto_markets(self) -> list:
        search_params = GetAssetsRequest(asset_class=AssetClass.CRYPTO)
        all_cryptos = self.client.get_all_assets(search_params)
        return all_cryptos

    def get_usd_crypto_markets(self) -> list:
        cryptos = self.get_all_crypto_markets()

        usd_markets = []
        for asset in cryptos:
            if 'US Dollar' in asset.name:
                usd_markets.append(asset)

        return usd_markets



