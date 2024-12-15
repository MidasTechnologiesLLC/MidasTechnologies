import os
import sys
import time
import json
import logging
import configparser
from datetime import datetime, timedelta
import csv
from decimal import Decimal
import pytz
import threading
import yfinance as yf 
import signal

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Color definitions for terminal output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"

# Configure logging with colors
def color_log(level, message, color):
    print(f"{color}[{level.upper()}]{Colors.RESET} {message}")

# Load config
config = configparser.ConfigParser()
if not os.path.exists('config.ini'):
    color_log("error", "config.ini not found. Please create one.", Colors.RED)
    sys.exit(1)

config.read('config.ini')
HOST = config.get('API', 'host', fallback='127.0.0.1')
PORT = config.getint('API', 'port', fallback=4002)
CLIENT_ID = config.getint('API', 'clientId', fallback=1)
DATA_DIR = config.get('Directories', 'data_dir', fallback='./data')
TICKER_FILE = config.get('General', 'ticker_file', fallback='ticker_ids.csv')
MIN_VOLUME = config.getint('Thresholds', 'min_volume', fallback=1000000)
MIN_NET_CHANGE = config.getfloat('Thresholds', 'min_net_change', fallback=0.1)
MIN_PERCENT_CHANGE = config.getfloat('Thresholds', 'min_percent_change', fallback=0.5)
MAX_SHARE_PRICE = config.getfloat('Thresholds', 'max_share_price', fallback=500.0)
GOOD_RUNTIME_THRESHOLD = config.getfloat('Performance', 'good_runtime_threshold', fallback=10.0)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(TICKER_FILE):
    color_log("error", f"{TICKER_FILE} not found in {DATA_DIR}. Please create one with a list of symbols.", Colors.RED)
    sys.exit(1)

# Debug loaded configuration
def debug_config():
    color_log("debug", "Configuration loaded:", Colors.BLUE)
    for section in config.sections():
        for key, value in config[section].items():
            color_log("debug", f"{section.upper()} - {key}: {value}", Colors.BLUE)

debug_config()

class IBWrapper(EWrapper):
    def __init__(self):
        super().__init__()
        self.current_time = None
        self.current_time_received = threading.Event()

        self.historical_data_reqId = None
        self.historical_bars = []
        self.historical_data_received = threading.Event()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in [2104, 2106, 2158]:
            color_log("info", f"System Check: ReqId {reqId}, Code {errorCode}, Msg: {errorString}", Colors.BLUE)
        else:
            color_log("error", f"Error. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}", Colors.RED)

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
        time.sleep(2)

    def disconnect_app(self):
        if self.isConnected():
            self.disconnect()

def calculate_start_date():
    """Calculate the start date for historical data (2 trading days ago)."""
    current_date = datetime.now()
    delta_days = 2
    while delta_days > 0:
        current_date -= timedelta(days=1)
        if current_date.weekday() < 5:  # Skip weekends
            delta_days -= 1
    return current_date.strftime("%Y%m%d")

def get_symbols_from_file(file_path):
    symbols = []
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            symbols.extend([symbol.strip() for symbol in row if symbol.strip()])
    return symbols

def request_historical_data(app, symbol, start_date):
    """Request historical data for a given symbol."""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"

    app.historical_bars = []
    app.historical_data_received.clear()
    app.historical_data_reqId = 10000  # arbitrary reqId

    endDateTime = f"{start_date} 09:30:00 UTC"
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
        color_log("error", f"Error requesting historical data for {symbol}: {e}", Colors.RED)
        return None

    if not app.historical_data_received.wait(timeout=10):
        color_log("warning", f"Timeout waiting for historical data for {symbol}.", Colors.YELLOW)
        return None

    return app.historical_bars

def filter_data(raw_data):
    """Filter raw data based on thresholds."""
    filtered_data = [
        entry for entry in raw_data
        if entry['volume'] >= MIN_VOLUME and
           entry['net_change'] >= MIN_NET_CHANGE and
           entry['percent_change'] >= MIN_PERCENT_CHANGE and
           entry['close'] <= MAX_SHARE_PRICE
    ]
    if not filtered_data:
        color_log("warning", "No data passed filtering. Check your config or provide a larger data pool.", Colors.YELLOW)
    return filtered_data

def refine_with_options(processed_data, app):
    """
    Refine processed data to include only stocks with associated option contracts.
    Attempts to use the IBKR API first, then falls back to Yahoo Finance.
    If IBKR fails persistently, it switches entirely to Yahoo Finance for subsequent checks.
    """
    refined_data = []
    ibkr_persistent_fail = False  # Flag to skip IBKR checks after persistent failures

    def check_with_ibkr(symbol):
        """Check if a stock has options contracts using IBKR."""
        nonlocal ibkr_persistent_fail

        if ibkr_persistent_fail:
            color_log("info", f"Skipping IBKR check for {symbol} due to persistent failures.", Colors.YELLOW)
            return False

        try:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"  # Stock type for the underlying security
            contract.exchange = "SMART"
            contract.currency = "USD"

            color_log("info", f"IBKR: Requesting options contract for {symbol} with contract parameters: {contract}", Colors.MAGENTA)

            app.historical_data_received.clear()
            app.reqSecDefOptParams(
                reqId=1,
                underlyingSymbol=symbol,
                futFopExchange="",
                underlyingSecType="STK",
                underlyingConId=0
            )

            if not app.historical_data_received.wait(timeout=5):  # Reduced timeout
                color_log("warning", f"IBKR: Timeout while querying options for {symbol}.", Colors.YELLOW)
                return False

            color_log("info", f"IBKR: Successfully queried options contract for {symbol}.", Colors.GREEN)
            return True
        except Exception as e:
            color_log("error", f"IBKR option check failed for {symbol}: {e}", Colors.RED)
            if "Invalid contract id" in str(e):  # Check for specific persistent error
                ibkr_persistent_fail = True
            return False

    def check_with_yfinance(symbol):
        """Check if a stock has options contracts using Yahoo Finance."""
        try:
            stock = yf.Ticker(symbol)
            if stock.options:
                color_log("info", f"Yahoo Finance: Options contract found for {symbol}.", Colors.GREEN)
                return True
            else:
                color_log("info", f"Yahoo Finance: No options contract found for {symbol}.", Colors.YELLOW)
                return False
        except Exception as e:
            color_log("warning", f"Yahoo Finance option check failed for {symbol}: {e}", Colors.YELLOW)
            return False

    # Process each stock in the data
    for entry in processed_data:
        symbol = entry['symbol']
        color_log("info", f"Checking options contracts for {symbol}...", Colors.MAGENTA)

        # Try IBKR first unless persistent failures are detected
        if check_with_ibkr(symbol):
            refined_data.append(entry)
            continue

        # Fallback to Yahoo Finance
        color_log("info", f"Falling back to Yahoo Finance for {symbol}...", Colors.BLUE)
        if check_with_yfinance(symbol):
            refined_data.append(entry)

    if not refined_data:
        color_log("warning", "No stocks with associated options contracts found.", Colors.YELLOW)

    return refined_data

def save_to_csv(ticker_ids, filename):
    """Save ticker IDs to a CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ticker_ids)
    color_log("info", f"Ticker IDs saved to {filename}.", Colors.GREEN)

def save_data(data, filename):
    """Save data to a JSON file."""
    if not data:
        color_log("warning", f"No data to save in {filename}.", Colors.YELLOW)
    else:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        color_log("info", f"Data saved to {filename}.", Colors.GREEN)

def handle_sigint(signal_received, frame):
    """
    Handle SIGINT (Ctrl+C) to perform cleanup before exiting.
    Deletes incomplete data files.
    """
    color_log("error", "SIGINT received. Cleaning up and exiting...", Colors.RED)

    # List of files to clean up
    temp_files = [
        os.path.join(DATA_DIR, f"raw_stock_info_{now_str}.json"),
        os.path.join(DATA_DIR, f"processed_data_{now_str}.json"),
        os.path.join(DATA_DIR, f"contract_option_stock_info_{now_str}.json")
    ]

    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)
            color_log("info", f"Deleted incomplete file: {file}", Colors.YELLOW)

    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, handle_sigint)
    start_time = time.time()
    app = IBApp()
    app.connect_app()

    if app.connect_error:
        color_log("error", f"Failed to connect to IB API: {app.connect_error}", Colors.RED)
        sys.exit(1)

    start_date = calculate_start_date()
    color_log("info", f"Start date for data retrieval: {start_date}", Colors.BLUE)

    symbols = get_symbols_from_file(TICKER_FILE)
    color_log("info", f"Loaded {len(symbols)} symbols from {TICKER_FILE}.", Colors.BLUE)

    raw_data = []
    ibkr_checks = 0
    yahoo_checks = 0

    for symbol in symbols:
        color_log("info", f"Retrieving data for {symbol}...", Colors.MAGENTA)
        bars = request_historical_data(app, symbol, start_date)
        if bars is None or len(bars) < 2:
            color_log("warning", f"Skipping {symbol}: not enough data.", Colors.YELLOW)
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
    save_data(raw_data, raw_filename)

    # Filter and save processed data
    processed_data = filter_data(raw_data)
    processed_filename = os.path.join(DATA_DIR, f"processed_data_{now_str}.json")
    save_data(processed_data, processed_filename)

    # Track fallback performance metrics
    color_log("info", "Starting refinement with options contracts...", Colors.BLUE)
    refined_data = refine_with_options(processed_data, app)
    refined_filename = os.path.join(DATA_DIR, f"contract_option_stock_info_{now_str}.json")
    save_data(refined_data, refined_filename)

    # Extract ticker IDs and save to CSV
    ticker_ids = [entry['symbol'] for entry in refined_data]
    csv_filename = os.path.join(DATA_DIR, "ticker_ids_with_match.csv")
    save_to_csv(ticker_ids, csv_filename)

    app.disconnect_app()
    end_time = time.time()
    runtime = end_time - start_time

    # Determine the proportion of IBKR and Yahoo Finance checks
    total_checks = len(processed_data)
    yahoo_checks = total_checks - ibkr_checks

    # Log detailed performance analysis
    if runtime < GOOD_RUNTIME_THRESHOLD:
        color_log("info", f"Program completed in {runtime:.2f} seconds (Good runtime).", Colors.GREEN)
    elif runtime == GOOD_RUNTIME_THRESHOLD:
        color_log("warning", f"Program completed in {runtime:.2f} seconds (Runtime threshold met).", Colors.YELLOW)
    else:
        color_log("error", f"Program completed in {runtime:.2f} seconds (Bad runtime).", Colors.RED)

    color_log("info", f"Refinement breakdown: IBKR checks = {ibkr_checks}, Yahoo Finance checks = {yahoo_checks}", Colors.MAGENTA)
    color_log(
        "info",
        "Time Complexity: O(k × t_ibkr + (n - k) × t_yahoo), where:\n"
        f"    k = {ibkr_checks} (IBKR checks)\n"
        f"    n = {total_checks} (Total symbols processed)\n"
        f"    t_ibkr = avg. IBKR query time\n"
        f"    t_yahoo = avg. Yahoo Finance query time",
        Colors.BLUE
    )

if __name__ == "__main__":
    main()
