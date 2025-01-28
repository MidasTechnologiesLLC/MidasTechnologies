import os
import sys
import time
import json
import shutil
import signal
import logging
import configparser
from datetime import datetime, timedelta
from decimal import Decimal
import pytz
import threading
import csv

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load config
config = configparser.ConfigParser()
if not os.path.exists('config.ini'):
    print("config.ini not found. Please create one.")
    sys.exit(1)

config.read('config.ini')
HOST = config.get('API', 'host', fallback='127.0.0.1')
PORT = config.getint('API', 'port', fallback=4002)
CLIENT_ID = config.getint('API', 'clientId', fallback=1)
DATA_DIR = config.get('Directories', 'data_dir', fallback='./data')
TICKER_FILE = config.get('General', 'ticker_file', fallback='ticker_ids.csv')
MIN_VOLUME = config.getint('Thresholds', 'min_volume', fallback=1000000)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(TICKER_FILE):
    print(f"{TICKER_FILE} not found. Please create one with a list of symbols.")
    sys.exit(1)

class IBWrapper(EWrapper):
    def __init__(self):
        super().__init__()
        self.current_time = None
        self.current_time_received = threading.Event()

        self.historical_data_reqId = None
        self.historical_bars = []
        self.historical_data_received = threading.Event()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        logging.error(f"Error. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}")

    def currentTime(self, time_):
        self.current_time = time_
        self.current_time_received.set()

    def historicalData(self, reqId, bar):
        if reqId == self.historical_data_reqId:
            self.historical_bars.append(bar)

    def historicalDataEnd(self, reqId, start, end):
        if reqId == self.historical_data_reqId:
            self.historical_data_received.set()

class IBClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)


class IBApp(IBWrapper, IBClient):
    def __init__(self):
        IBWrapper.__init__(self)
        IBClient.__init__(self, wrapper=self)
        self.connect_error = None

    def connect_app(self):
        try:
            self.connect(HOST, PORT, CLIENT_ID)
        except Exception as e:
            self.connect_error = e

        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        # Wait a bit for connection
        time.sleep(2)

    def disconnect_app(self):
        if self.isConnected():
            self.disconnect()

def signal_handler(sig, frame):
    print("Interrupt received, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def is_trading_hours(now_eastern):
    # Trading hours: Monday-Friday, 9:30 - 16:00 Eastern
    if now_eastern.weekday() > 4:  # Saturday=5, Sunday=6
        return False
    open_time = now_eastern.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now_eastern.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now_eastern <= close_time

def get_symbols_from_file(file_path):
    symbols = []
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            # Extend the symbols list with all symbols in the current row
            symbols.extend([symbol.strip() for symbol in row if symbol.strip()])
    return symbols


def request_historical_data(app, symbol):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"

    app.historical_bars = []
    app.historical_data_received.clear()
    app.historical_data_reqId = 10000  # arbitrary reqId

    endDateTime = datetime.now().strftime("%Y%m%d %H:%M:%S") + " UTC"
    try:
        app.reqHistoricalData(
            reqId=app.historical_data_reqId,
            contract=contract,
            endDateTime=endDateTime,
            durationStr="2 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
    except Exception as e:
        logging.error(f"Error requesting historical data for {symbol}: {e}")
        return None

    if not app.historical_data_received.wait(timeout=10):
        logging.warning(f"Timeout waiting for historical data for {symbol}.")
        return None

    return app.historical_bars

def serialize_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def main():
    app = IBApp()
    app.connect_app()

    if app.connect_error:
        logging.error(f"Failed to connect to IB API: {app.connect_error}")
        sys.exit(1)

    app.reqCurrentTime()
    if not app.current_time_received.wait(timeout=10):
        logging.error("Timeout waiting for current time.")
        app.disconnect_app()
        sys.exit(1)

    ib_server_time = datetime.utcfromtimestamp(app.current_time)
    eastern = pytz.timezone("US/Eastern")
    ib_server_time_eastern = ib_server_time.replace(tzinfo=pytz.utc).astimezone(eastern)

    if is_trading_hours(ib_server_time_eastern):
        logging.info("It's currently within trading hours. No historical retrieval needed.")
        app.disconnect_app()
        sys.exit(0)
    else:
        logging.info("Outside trading hours. Proceeding with historical data retrieval.")

    symbols = get_symbols_from_file(TICKER_FILE)
    logging.info(f"Loaded {len(symbols)} symbols from {TICKER_FILE}.")

    raw_data = []
    for symbol in symbols:
        bars = request_historical_data(app, symbol)
        if bars is None or len(bars) < 2:
            logging.info(f"Skipping {symbol}: not enough data.")
        else:
            last_bar = bars[-1]
            prev_bar = bars[-2]
            net_change = last_bar.close - prev_bar.close
            percent_change = (net_change / prev_bar.close) * 100 if prev_bar.close != 0 else 0.0

            entry = {
                "symbol": symbol,
                "date": last_bar.date,
                "open": float(last_bar.open),
                "high": float(last_bar.high),
                "low": float(last_bar.low),
                "close": float(last_bar.close),
                "volume": int(last_bar.volume),
                "net_change": float(net_change),
                "percent_change": float(percent_change)
            }
            raw_data.append(entry)

        time.sleep(0.5)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_filename = os.path.join(DATA_DIR, f"raw_stock_info_{now_str}.json")
    with open(raw_filename, 'w') as f:
        json.dump(raw_data, f, indent=4, default=serialize_decimal)
    logging.info(f"Raw data saved to {raw_filename}")

    app.disconnect_app()
    logging.info("All done.")

if __name__ == "__main__":
    main()

