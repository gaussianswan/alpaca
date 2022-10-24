from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import logging
import asyncio
import os
import datetime
from dotenv import load_dotenv

# Setting up the logger
# ENABLE LOGGING
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename = 'neural_net_strat.log',
    filemode='w'
)

logger = logging.getLogger(__name__)

# Creating the alpaca trading client
dotenv_path = r'C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca\.env'
load_dotenv(dotenv_path = dotenv_path)
api_key = os.environ['ALPACA_PAPER_TRADING_KEY_ID']
secret_key = os.environ['ALPACA_PAPER_TRADING_SECRET_KEY']
trading_client = TradingClient(api_key = api_key, secret_key=secret_key)
waiting_time = 60  # we are waiting 60 seconds until we do another call

class LSTMStockStrategy:
    """Class for predicting price series and running
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        trading_pair: str = 'ETH/USD',
        feature: str = 'close',
        look_back: int = 200,
        timeframe: TimeFrame = TimeFrame.Minute,
        neurons: int = 50,
        activ_func: str = 'linear',
        dropout: float = 0.2,
        loss: str = 'mse',
        optimizer: str = 'adam',
        epochs: int = 20,
        batch_size: int = 32,
        output_size: int = 1,
        scaler_type = MinMaxScaler
    ) -> None:
        self.feature = feature
        self.look_back = look_back
        self.neurons = neurons
        self.activ_func = activ_func
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.timeframe = timeframe
        self.batch_size = batch_size
        self.output_size = output_size
        self.trading_pair = trading_pair
        self.api_key = api_key
        self.secret_key = secret_key
        self.candle_look_back = 1000
        self.historical_data_client = CryptoHistoricalDataClient()
        self.trading_client = TradingClient(api_key = self.api_key, secret_key=self.secret_key)
        self.scaler_type = scaler_type

    def get_historical_data(self):

        time_diff = datetime.datetime.now() - relativedelta(minutes = self.candle_look_back)

        logger.info("Getting bar data for {} start from {}".format(self.trading_pair, time_diff))

        request_params = CryptoBarsRequest(
            symbol_or_symbols=[self.trading_pair],
            timeframe = self.timeframe,
            start = time_diff
        )

        df = self.historical_data_client.get_crypto_bars(request_params=request_params).df
        return df

    def get_feature(self, df):

        data = df.filter([self.feature])
        data = data.values
        return data

    def scale_data(self, data):

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    def get_train_data(self, scaled_data):
        x, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            x.append(scaled_data[i - self.look_back:i, :])
            y.append(scaled_data[i, :])

        return np.array(x), np.array(y)

    def LSTM_model(self, input_data):
        """Creating the LSTM model and returning it

        Args:
            input_data (_type_): _description_

        Returns:
            _type_: _description_
        """

        # We are creating a sequential model with three LSTM layers with dropout
        model = Sequential()
        model.add(LSTM(self.neurons, input_shape=(
            input_data.shape[1], input_data.shape[2]), return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons))
        model.add(Dropout(self.dropout))

        # The dense layer here has the same shape as the output layer which should be a single prediction
        model.add(Dense(units=self.output_size))
        model.add(Activation(self.activ_func))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def train_model(self, x, y):
        x_train = x[:len(x) - 1]
        y_train = y[:len(x) - 1]
        model = self.LSTM_model(x_train)
        modelfit = model.fit(x_train, y_train, epochs = self.epochs, batch_size = self.batch_size, verbose = 1, shuffle = True)
        return model, modelfit

    def predict(self):
        logging.info("Getting bar data")

        df = self.get_historical_data()

        logger.info("Getting Feature: {}".format(self.feature))

        data = self.get_feature(df)

        logger.info("Scaling data...")
        scaled_data, scaler = self.scale_data(data)

        logger.info("Getting Train Data")
        x_train, y_train = self.get_train_data(scaled_data)

        logger.info("Training Model")
        model = self.train_model(x_train, y_train)[0]

        logger.info("Extracting data to predict on")
        x_pred = scaled_data[-self.look_back]
        x_pred = np.reshape(x_pred, (1, x_pred.shape[0]))


        # Predict the result
        logger.info("Predicting Price")
        pred = model.predict(x_pred).squeeze()
        pred = np.array([float(pred)])
        pred = np.reshape(pred, (pred.shape[0], 1))

        # Inverse the scaling to get the actual price
        pred_true = scaler.inverse_transform(pred)
        return pred_true[0][0]

def get_positions():
    positions = trading_client.get_all_positions()
    global current_position
    for p in positions:
        if p.symbol == 'ETHUSD':
            current_position = p.qty
            return current_position
    return current_position

async def post_alpaca_order(side: str, quantity: int):

    try:
        if side == 'buy':
            market_order_data = MarketOrderRequest(
                symbol = 'ETHUSD',
                qty = quantity,
                side = OrderSide.BUY,
                time_in_force = TimeInForce.GTC
            )

            buy_order = trading_client.submit_order(
                order_data = market_order_data
            )

            return buy_order

        else:
            market_order_data = MarketOrderRequest(
                symbol = 'ETHUSD',
                qty = current_position,
                side = OrderSide.SELL,
                time_in_force = TimeInForce.GTC
            )

            sell_order = trading_client.submit_order(
                order_data = market_order_data
            )

            return sell_order

    except Exception as e:
        logger.exception(
            "There was an issue posting order to Alpaca: {0}".format(e)
        )

        return False


async def check_condition():

    global current_position, current_price, predicted_price
    current_position = get_positions()

    logger.info("Current Price is {0}".format(current_price))
    logger.info("Current Position is: {0}".format(current_position))

    if float(current_position) <= 0.01 and current_price < predicted_price:
        logger.info("Placing buy order")
        buy_order = await post_alpaca_order('buy', quantity = 5)

        if buy_order:
            logger.info("Buy Order Placed!")

    if float(current_position) >= 0.01 and current_price > predicted_price:
        logger.info("Placing Sell Order")
        sell_order = await post_alpaca_order("sell", float(current_position))

        if sell_order:
            logging.info("Sell Order Placed!")

async def main():

    while True:
        logger.info("-----------------------")
        pred = LSTMStockStrategy(api_key=api_key, secret_key=secret_key)
        global predicted_price
        predicted_price = pred.predict()
        logger.info("Predicted Price is {0}".format(predicted_price))
        l1 = loop.create_task(check_condition())
        await asyncio.wait([l1])
        await asyncio.sleep(waiting_time)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()




