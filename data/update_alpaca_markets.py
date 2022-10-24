
import os
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from alpaca.trading.models import Asset
from dotenv import load_dotenv


dotenv_path = r'C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca\.env'
load_dotenv(dotenv_path=dotenv_path)

api_key = os.environ['ALPACA_PAPER_TRADING_KEY_ID']
secret_key = os.environ['ALPACA_PAPER_TRADING_SECRET_KEY']

def filter_asset(asset: Asset) -> bool:

    if asset.tradable and asset.shortable and asset.easy_to_borrow and asset.marginable:
        return True
    else:
        return False

def unpack_asset(asset: Asset) -> list:
    unpacked = [
        asset.symbol,
        asset.asset_class.name,
        asset.exchange.name,
        asset.fractionable,
        str(asset.id),
        asset.marginable,
        asset.name,
        asset.shortable,
        asset.status.name,
        asset.tradable
    ]

    return unpacked

if __name__ == '__main__':
    trading_client = TradingClient(api_key=api_key, secret_key = secret_key)
    search_equity_assets = GetAssetsRequest(asset_class = AssetClass.US_EQUITY)
    assets = trading_client.get_all_assets(search_equity_assets)

    # Applying the filter to these assets:
    desired_assets = list(filter(filter_asset, assets))

    assets_unpacked = []
    for asset in desired_assets:
        assets_unpacked.append(unpack_asset(asset))

    assets_df = pd.DataFrame(assets_unpacked)
    assets_df.columns = ['SYMBOL', 'ASSET_CLASS', 'EXCHANGE', 'FRACTIONABLE', 'ID', 'MARGINABLE', 'NAME', 'SHORTABLE', 'STATUS', 'TRADABLE']

    assets_df.to_csv(r"C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca\data\alpaca_equity_markets.csv")



