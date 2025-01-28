# test_stock_retriever.py

"""
========================================================================
# README
#
# Program: test_stock_retriever.py
#
# Description:
# This script:
# 1) Connects to IB Gateway or TWS.
# 2) Requests and displays the current IB server date/time.
# 3) Loads a list of stock symbols from a local file 'symbols.txt' (one symbol per line).
# 4) Shows which symbols will be queried.
# 5) Requests historical data (last day's OHLCV) for these symbols sequentially.
# 6) Waits until all symbols are processed or times out.
# 7) Prints out the close price (as last price) and volume for each symbol if available,
#    or indicates no data.
#
# Usage:
#   python test_stock_retriever.py
#
# Ensure that 'symbols.txt' is in the same directory as this script.
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

# ANSI color codes for enhanced console outputs
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
BLUE  = "\033[94m"
MAGENTA="\033[95m"
RESET = "\033[0m"

# Known "info" codes that should not be treated as errors
INFO_CODES = {2104, 2106, 2107, 2158, 2159}

logging.basicConfig(
    filename='test_stock_retriever.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TestStockRetriever(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.stock_data = {}     # symbol -> {'last_price': float, 'volume': int}
        self.data_event = threading.Event()
        self.logger = logging.getLogger('MidasV1.TestStockRetriever')

        self.current_server_time = None
        self.symbols = {}          # {reqId: symbol}
        self.symbols_remaining = 0 # How many symbols left to process?

        # For historical data retrieval
        self.current_req_id = None
        self.current_symbol = None
        self.current_bars = []

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in INFO_CODES:
            print(f"{YELLOW}Info. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}{RESET}")
            logging.info(f"Info. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}")
        else:
            print(f"{RED}Error. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}{RESET}")
            logging.error(f"Error. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}, Advanced JSON: {advancedOrderRejectJson}")

    def currentTime(self, time_: int):
        self.current_server_time = time_
        self.logger.info(f"Current IB Server Time: {time_}")
        print(f"{BLUE}Current IB Server Time (Epoch): {time_}{RESET}")
        self.after_time_retrieved()

    def after_time_retrieved(self):
        # Convert epoch time to human-readable
        local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.current_server_time))
        print(f"{GREEN}Current IB Server Date/Time: {local_time_str}{RESET}")

        # Load the symbol list from a file
        loaded_symbol_list = self.load_symbols_from_file('symbols.txt')
        print(f"{MAGENTA}The following symbols have been loaded from symbols.txt for {local_time_str}:{RESET}")
        for s in loaded_symbol_list:
            print(f" - {s}")

        # Assign ReqId to each symbol
        self.symbols = {1001 + idx: sym for idx, sym in enumerate(loaded_symbol_list)}
        self.symbols_remaining = len(self.symbols)

        # Now we can request historical data
        self.request_market_data()

    def load_symbols_from_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            return symbols
        except FileNotFoundError:
            print(f"{RED}Error: {file_path} not found! Make sure your symbol file is available.{RESET}")
            self.disconnect()
            sys.exit(1)

    def nextValidId(self, orderId):
        logging.info(f"NextValidId received: {orderId}")
        self.reqCurrentTime()

    def request_market_data(self):
        logging.info("Requesting historical market data for symbols...")
        print(f"{BLUE}\nRequesting historical market data for the following symbols:{RESET}")
        for reqId, symbol in self.symbols.items():
            print(f" - {symbol} (ReqId: {reqId})")
        print()

        # Fetch each symbol sequentially in a separate thread
        threading.Thread(target=self.fetch_all_symbols_data, daemon=True).start()

    def fetch_all_symbols_data(self):
        for reqId, symbol in self.symbols.items():
            self.request_symbol_historical_data(reqId, symbol)
        # Once all symbols are done, set the event
        self.data_event.set()

    def request_symbol_historical_data(self, reqId, symbol):
        self.current_req_id = reqId
        self.current_symbol = symbol
        self.current_bars = []

        # We'll request one day of daily bars ending now
        endDateTime = time.strftime("%Y%m%d %H:%M:%S", time.localtime(time.time()))
        durationStr = "1 D"
        barSizeSetting = "1 day"
        whatToShow = "TRADES"

        self.reqHistoricalData(
            reqId=reqId,
            contract=self.create_contract(symbol),
            endDateTime=endDateTime + " UTC",
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow=whatToShow,
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

        # Wait a short time for data (or no data)
        # We'll rely on historicalDataEnd to know when data ends.
        # If no data arrives, historicalDataEnd should still fire quickly.
        start_time = time.time()
        timeout = 10
        while ((time.time() - start_time) < timeout) and (self.current_req_id is not None):
            time.sleep(0.1)

        # If still current_req_id not None, means no historicalDataEnd arrived or delayed
        # We'll just proceed. The symbol will show no data available if historicalDataEnd didn't fire.

    def create_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def historicalData(self, reqId, bar):
        if reqId == self.current_req_id and self.current_symbol:
            self.current_bars.append(bar)

    def historicalDataEnd(self, reqId, start, end):
        # When historical data ends for this symbol:
        if reqId in self.symbols:
            symbol = self.symbols[reqId]
            if self.current_bars:
                last_bar = self.current_bars[-1]
                self.stock_data[symbol] = {
                    'last_price': last_bar.close,
                    'volume': last_bar.volume
                }
                print(f"{GREEN}Received data for {symbol}: Close=${last_bar.close}, Volume={last_bar.volume}{RESET}")
            else:
                print(f"{RED}No Data Available for {symbol}.{RESET}")

            # Mark this symbol as done
            self.current_req_id = None
            self.current_symbol = None
            self.current_bars = []

            self.symbols_remaining -= 1

    def connectionClosed(self):
        logging.info("Connection to IB Gateway closed.")
        print(f"{YELLOW}Connection to IB Gateway closed.{RESET}")

    def run_retriever(self):
        # Start the socket in a thread
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()

        # Wait for all symbols to finish or timeout after 60 seconds
        if self.data_event.wait(timeout=60):
            logging.info("All historical market data retrieval completed.")
            print(f"{GREEN}Historical market data retrieval completed.{RESET}")
        else:
            logging.warning("Timeout while waiting for all historical market data.")
            print(f"{RED}Timeout while waiting for historical market data.{RESET}")

        # Disconnect after data retrieval
        self.disconnect()

    def stop(self):
        self.disconnect()

def signal_handler(sig, frame, app):
    logger = logging.getLogger('MidasV1.TestStockRetriever')
    logger.error("Interrupt received. Shutting down gracefully...")
    print(f"{RED}Interrupt received. Shutting down gracefully...{RESET}")
    app.disconnect()
    sys.exit(0)

def main():
    app = TestStockRetriever()

    # Register the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, app))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, app))

    # Connect to IB Gateway or TWS
    app.connect("127.0.0.1", 4002, clientId=1)

    # Run the retriever
    app.run_retriever()

    # Allow some time for data to be processed
    time.sleep(2)

    # Print final results for each symbol
    print(f"{BLUE}\nFinal Results:{RESET}")
    print("-----------------------------")
    for reqId, symbol in app.symbols.items():
        data = app.stock_data.get(symbol, {})
        last_price = data.get('last_price', None)
        volume = data.get('volume', None)

        if last_price is not None and volume is not None:
            print(f"Symbol: {symbol}, Last Price: ${last_price:.2f}, Volume: {volume}")
        else:
            print(f"Symbol: {symbol}, No Data Available")

    print("-----------------------------\n")
    logging.info("TestStockRetriever completed.")

if __name__ == "__main__":
    main()

