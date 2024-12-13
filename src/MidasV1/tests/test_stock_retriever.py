# test_stock_retriever.py

"""
========================================================================
# README
#
# Program: test_stock_retriever.py
#
# Description:
# This script serves as a testing tool for the MidasV1 Trading Bot.
# It connects to the Interactive Brokers (IB) Gateway, retrieves real-time
# market data for a predefined list of stock symbols, and applies internal
# criteria to filter the stocks based on share price and trading volume.
#
# Features:
# - Connects to IB Gateway on a specified host and port.
# - Requests market data (last price and volume) for a list of stock symbols.
# - Applies filtering criteria to identify stocks that meet minimum share price
#   and trading volume requirements.
# - Provides colored console outputs for better readability and user feedback.
# - Handles graceful shutdown on interrupt signals and error conditions.
# - Implements robust error handling to manage API connection issues and data retrieval problems.
#
# Usage:
# Run the script from the command line:
#   python test_stock_retriever.py
#
# The script will attempt to connect to IB Gateway, request market data, and display
# the list of stocks that meet the specified criteria. If no stocks meet the criteria,
# it will notify the user accordingly.
#
# Enhancements:
# - Added colored outputs using ANSI escape codes (green for pass, red for fail).
# - Implemented signal handling for graceful exits on interrupts.
# - Enhanced error handling for various API and connection errors.
# - Improved logging for detailed traceability.
#
# Coded by: kleinpanic 2024
========================================================================
"""

import logging
import threading
import time
import signal
import sys

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum

# ANSI color codes for enhanced console outputs
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# Configure logging
logging.basicConfig(
    filename='test_stock_retriever.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TestStockRetriever(EWrapper, EClient):
    def __init__(self, symbols, criteria):
        EClient.__init__(self, self)
        self.symbols = symbols
        self.criteria = criteria
        self.stock_data = {}
        self.data_event = threading.Event()
        self.logger = logging.getLogger('MidasV1.TestStockRetriever')

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        logging.error(f"Error. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}, Advanced Order Reject JSON: {advancedOrderRejectJson}")
        print(f"{RED}Error. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}, Advanced Order Reject JSON: {advancedOrderRejectJson}{RESET}")

    def tickPrice(self, reqId, tickType, price, attrib):
        # We are interested in LAST price
        if tickType == TickTypeEnum.LAST:
            symbol = self.symbols.get(reqId, None)
            if symbol:
                self.stock_data.setdefault(symbol, {})['last_price'] = price
                logging.debug(f"TickPrice. ReqId: {reqId}, Symbol: {symbol}, Last Price: {price}")
                # Check if both last_price and volume are received
                if 'volume' in self.stock_data[symbol]:
                    self.data_event.set()
                    print(f"{GREEN}Received last price for {symbol}: ${price}{RESET}")

    def tickSize(self, reqId, tickType, size):
        # We are interested in VOLUME
        if tickType == TickTypeEnum.VOLUME:
            symbol = self.symbols.get(reqId, None)
            if symbol:
                self.stock_data.setdefault(symbol, {})['volume'] = size
                logging.debug(f"TickSize. ReqId: {reqId}, Symbol: {symbol}, Volume: {size}")
                # Check if both last_price and volume are received
                if 'last_price' in self.stock_data[symbol]:
                    self.data_event.set()
                    print(f"{GREEN}Received volume for {symbol}: {size}{RESET}")

    def tickString(self, reqId, tickType, value):
        # Optionally handle other tick types
        pass

    def tickGeneric(self, reqId, tickType, value):
        # Optionally handle other tick types
        pass

    def nextValidId(self, orderId):
        # Start requesting market data once the next valid order ID is received
        logging.info(f"NextValidId received: {orderId}")
        self.request_market_data()

    def request_market_data(self):
        logging.info("Requesting market data for symbols...")
        print(f"{BLUE}Requesting market data for symbols...{RESET}")
        for reqId, symbol in self.symbols.items():
            contract = self.create_contract(symbol)
            self.reqMktData(reqId, contract, "", False, False, [])
            logging.debug(f"Requested market data for {symbol} with ReqId: {reqId}")
            print(f"{BLUE}Requested market data for {symbol} with ReqId: {reqId}{RESET}")

    def create_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def marketDataType(self, reqId, marketDataType):
        # Optionally handle different market data types
        logging.debug(f"MarketDataType. ReqId: {reqId}, Type: {marketDataType}")

    def connectionClosed(self):
        logging.info("Connection to IB Gateway closed.")
        print(f"{YELLOW}Connection to IB Gateway closed.{RESET}")

    def run_retriever(self):
        # Start the socket in a thread
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()

        # Wait for data to be collected or timeout after 10 seconds
        if self.data_event.wait(timeout=10):
            logging.info("Market data retrieved successfully.")
            print(f"{GREEN}Market data retrieved successfully.{RESET}")
        else:
            logging.warning("Timeout while waiting for market data.")
            print(f"{RED}Timeout while waiting for market data.{RESET}")

        # Disconnect after data retrieval
        self.disconnect()

    def stop(self):
        self.disconnect()

def signal_handler(sig, frame, app):
    """
    Handles incoming signals for graceful shutdown.

    Args:
        sig (int): Signal number.
        frame: Current stack frame.
        app (TestStockRetriever): The running application instance.
    """
    logger = logging.getLogger('MidasV1.TestStockRetriever')
    logger.error("Interrupt received. Shutting down gracefully...")
    print(f"{RED}Interrupt received. Shutting down gracefully...{RESET}")
    app.disconnect()
    sys.exit(0)

def main():
    # Define a list of stock symbols to retrieve data for
    symbols_to_test = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    # Assign a unique ReqId to each symbol
    symbols = {1001 + idx: symbol for idx, symbol in enumerate(symbols_to_test)}

    # Define internal criteria for filtering stocks
    criteria = {
        'min_share_price': 50.0,   # Minimum share price in USD
        'min_volume': 1000000      # Minimum trading volume
    }

    # Instantiate the TestStockRetriever
    app = TestStockRetriever(symbols, criteria)

    # Register the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, app))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, app))

    # Connect to IB Gateway or TWS
    # Update the port and clientId based on your setup
    app.connect("127.0.0.1", 4002, clientId=1)  # Use 4002 for IB Gateway Simulated Trading

    # Start the data retrieval process
    app.run_retriever()

    # Allow some time for data to be processed
    time.sleep(2)

    # Apply criteria to filter stocks
    filtered_stocks = []
    for symbol, data in app.stock_data.items():
        last_price = data.get('last_price', 0)
        volume = data.get('volume', 0)
        if last_price >= criteria['min_share_price'] and volume >= criteria['min_volume']:
            filtered_stocks.append({
                'symbol': symbol,
                'last_price': last_price,
                'volume': volume
            })
            logging.info(f"Stock {symbol} meets criteria: Price=${last_price}, Volume={volume}")
        else:
            logging.info(f"Stock {symbol} does not meet criteria: Price=${last_price}, Volume={volume}")

    # Display the filtered stocks
    if filtered_stocks:
        print(f"{GREEN}\nStocks meeting the criteria:{RESET}")
        print("-----------------------------")
        for stock in filtered_stocks:
            print(f"Symbol: {stock['symbol']}, Last Price: ${stock['last_price']:.2f}, Volume: {stock['volume']}")
        print("-----------------------------\n")
    else:
        print(f"{RED}\nNo stocks met the specified criteria.\n{RESET}")

    # Log the end of the test
    logging.info("TestStockRetriever completed.")

if __name__ == "__main__":
    main()

