from alpaca.data import CryptoDataStream, StockDataStream
from load_api_keys import load_alpaca_paper_trading_keys

api_keys = load_alpaca_paper_trading_keys()
crypto_stream = CryptoDataStream(api_key=api_keys.api_key, secret_key=api_keys.secret_key)

async def quote_data_handler(data):
    print(type(data))
    print(data)

crypto_stream.subscribe_quotes(quote_data_handler, symbols=['ETH/USD', 'BTC/USD'])
crypto_stream.run()