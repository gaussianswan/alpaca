from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import json
import logging
import asyncio
import os
import datetime
from dotenv import load_dotenv

# ENABLE LOGGING
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Creating the alpaca trading client
dotenv_path = r'C:\Users\srerr\Documents\Projects\PersonalProjects\stonks\alpaca\.env'
load_dotenv(dotenv_path = dotenv_path)
api_key = os.environ['ALPACA_PAPER_TRADING_KEY_ID']
secret_key = os.environ['ALPACA_PAPER_TRADING_SECRET_KEY']
trading_client = TradingClient(api_key = api_key, secret_key=secret_key)

# Defining the trading variables
trading_pair = 'ETH/USD'
quantity = 5

waitTime = 3600
data = 0
current_position, current_price = 0, 0
predicted_price = 0

class LSTMStockPredictionModel:
    """Class for predicting price series
    """

    def __init__(self,
                past_days: int = 50,
                trading_pair: str = 'ETHUSD',
                exchange: str = 'FTXU',
                feature: str = 'close',
                look_back: int = 100,
                neurons: int = 50,
                activ_func: str = 'linear',
                dropout: float = 0.2,
                loss: str = 'mse',
                optimizer: str = 'adam',
                epochs: int = 20,
                batch_size: int = 32,
                output_size: int = 1
                ):
        self.exchange = exchange
        self.feature = feature
        self.look_back = look_back
        self.neurons = neurons
        self.activ_func = activ_func
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_size = output_size

    def getAllData(self):
        """Retrieves data from alpaca for the past 1000 hours and returns the dataframe

        Returns:
            _type_: _description_
        """
        data_client = CryptoHistoricalDataClient()

        time_diff = datetime.now() - relativedelta(hours=1000)
        logger.info("Getting bar data for {0} starting from {1}".format(trading_pair, time_diff))
        # Defining Bar data request parameters
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[trading_pair],
            timeframe=TimeFrame.Hour,
            start=time_diff)

        # Get the bar data from Alpaca
        df = data_client.get_crypto_bars(request_params).df
        global current_price
        current_price = df.iloc[-1]['close']
        return df

    def getFeature(self, df):
        """Gets a specific feature from the dataframe you provide

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """
        data = df.filter([self.feature])
        data = data.values
        return data

    def scaleData(self, data):
        """Scales the data using a scaler of your choice

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    # train on all data for which labels are available (train + test from dev)
    def getTrainData(self, scaled_data):
        x, y = [], []
        for price in range(self.look_back, len(scaled_data)):
            x.append(scaled_data[price - self.look_back:price, :])
            y.append(scaled_data[price, :])

        return np.array(x), np.array(y)

    def LSTM_model(self, input_data):
        model = Sequential()
        model.add(LSTM(self.neurons, input_shape=(
            input_data.shape[1], input_data.shape[2]), return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=self.output_size))
        model.add(Activation(self.activ_func))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def trainModel(self, x, y):
       x_train = x[: len(x) - 1]
       y_train = y[: len(x) - 1]
       model = self.LSTM_model(x_train)
       modelfit = model.fit(x_train, y_train, epochs=self.epochs,
                            batch_size=self.batch_size, verbose=1, shuffle=True)
       return model, modelfit

    def predictModel(self):
        logger.info("Getting Ethereum Bar Data")
        df = self.getAllData()


        logger.info("Getting Feature: {}".format(self.feature))
        data = self.getFeature(df)

        logger.info("Scaling data...")
        scaled_data, scaler = self.scaleData(data)

        logger.info("Getting Train Data")
        x_train, y_train = self.getTrainData(scaled_data)

        logger.info("Training Model")
        model = self.trainModel(x_train, y_train)[0]

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

async def post_alpaca_order(side: str):

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
        buy_order = await post_alpaca_order('buy')

        if buy_order:
            logger.info("Buy Order Placed!")

    if float(current_position) >= 0.01 and current_price > predicted_price:
        logger.info("Placing Sell Order")
        sell_order = await post_alpaca_order("sell")

        if sell_order:
            logging.info("Sell Order Placed!")

async def main():

    while True:
        logger.info("-----------------------")
        pred = LSTMStockPredictionModel()
        global predicted_price
        predicted_price = pred.predictModel()
        logger.info("Predicted Price is {0}".format(predicted_price))
        l1 = loop.create_task(check_condition())
        await asyncio.wait([l1])
        await asyncio.sleep(waitTime)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()