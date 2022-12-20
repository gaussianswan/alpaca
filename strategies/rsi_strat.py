from meanreversion import CryptoLongRSIStrategy
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from load_api_keys import load_alpaca_paper_trading_keys, load_telegram_bot_keys

import sys
import pathlib
import os
import pickle

sys.path.append(pathlib.Path(r'C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca'))


dotenv_path = r'C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca\.env'
symbol = "ETH/USD"
timeframe = TimeFrame(5, TimeFrameUnit.Minute)
capital = 1000

if __name__ == "__main__":
    alpaca_paper_trading_keys = load_alpaca_paper_trading_keys(dotenv_path=dotenv_path)
    telegram_bot_keys = load_telegram_bot_keys(dotenv_path=dotenv_path)

    entry_rsi_value = 30
    exit_rsi_value = 70
    rsi_lookback_period = 14

    strategy_name = "RSI_strategy"
    directory = pathlib.Path(r'C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca\strategies\livestrats')
    pkl_file_path = directory / "RSI_strategy.pkl"

    if os.path.exists(pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            strat = pickle.load(f)
    else:
        strat = CryptoLongRSIStrategy(
        symbol=symbol,
        timeframe=timeframe,
        alpaca_paper_trading_keys=alpaca_paper_trading_keys,
        telegram_bot_keys=telegram_bot_keys,
        capital=capital,
        entry_rsi_value=entry_rsi_value,
        exit_rsi_value=exit_rsi_value,
        rsi_lookback_period=rsi_lookback_period
    )

    # Then we are going to run the strategy
    strat.run()

    # After running, we can save this down
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(strat, f)




