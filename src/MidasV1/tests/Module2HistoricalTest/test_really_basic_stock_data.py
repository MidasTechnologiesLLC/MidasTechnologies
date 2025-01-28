# test_really_basic_stock_data.py

import logging
import threading
import time
import signal
import sys

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# ANSI colors for clarity
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# Known info codes that should be printed as informational messages (not errors)
INFO_CODES = {2104, 2106, 2107, 2158, 2159}

logging.basicConfig(
    filename='test_really_basic_stock_data.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SymbolInfoRetriever(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.logger = logging.getLogger('MidasV1.SymbolInfoRetriever')

        self.details_event = threading.Event()
        self.contract_details = {}  # symbol -> list of contract details
        self.symbols = {}
        self.details_requested = 0
        self.details_received = 0

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Print informational codes in yellow, actual errors in red
        if errorCode in INFO_CODES:
            print(f"{YELLOW}Info. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}{RESET}")
            logging.info(f"Info. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}")
        else:
            print(f"{RED}Error. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}{RESET}")
            logging.error(f"Error. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}")

    def nextValidId(self, orderId):
        # Once we have a valid ID, we can proceed with requesting symbol info
        self.request_symbol_info()

    def request_symbol_info(self):
        symbol_list = self.load_symbols('symbols.txt')
        print(f"{MAGENTA}Loaded symbols:{RESET}")
        for s in symbol_list:
            print(f" - {s}")

        self.symbols = {1001 + i: s for i, s in enumerate(symbol_list)}
        self.details_requested = len(symbol_list)

        print(f"{BLUE}\nRequesting contract details for the following symbols:{RESET}")
        for reqId, symbol in self.symbols.items():
            print(f" - {symbol} (ReqId: {reqId})")
        print()

        for reqId, symbol in self.symbols.items():
            contract = self.create_contract(symbol)
            self.reqContractDetails(reqId, contract)

    def create_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def contractDetails(self, reqId, contractDetails):
        symbol = self.symbols.get(reqId, None)
        if symbol:
            self.contract_details.setdefault(symbol, []).append(contractDetails)

    def contractDetailsEnd(self, reqId):
        self.details_received += 1
        if self.details_received == self.details_requested:
            self.details_event.set()

    def connectionClosed(self):
        print(f"{YELLOW}Connection to IB Gateway closed.{RESET}")

    def run_retriever(self):
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()

        # Wait for contract details or timeout
        if self.details_event.wait(timeout=10):
            print(f"{GREEN}Contract details retrieval completed.{RESET}")
        else:
            print(f"{RED}Timeout waiting for contract details.{RESET}")

        self.disconnect()

    def load_symbols(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"{RED}Error: {file_path} not found! Make sure your symbol file is available.{RESET}")
            self.disconnect()
            sys.exit(1)

    def stop(self):
        self.disconnect()

def signal_handler(sig, frame, app):
    print(f"{RED}Interrupt received. Shutting down...{RESET}")
    app.disconnect()
    sys.exit(0)

def main():
    app = SymbolInfoRetriever()

    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, app))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, app))

    app.connect("127.0.0.1", 4002, clientId=1)
    app.run_retriever()

    # Print the retrieved contract details in a nicer format
    print(f"{BLUE}\n========== CONTRACT DETAILS RESULTS =========={RESET}")
    if not app.contract_details:
        print(f"{RED}No contract details received at all.{RESET}")
    else:
        for symbol, details_list in app.contract_details.items():
            if not details_list:
                print(f"{YELLOW}\nSymbol: {symbol}, No Contract Details Received{RESET}")
                continue

            # Usually only one primary detail set is returned, but we loop in case there are multiple.
            for d in details_list:
                print(f"{MAGENTA}\n--------------------------------------------{RESET}")
                print(f"{GREEN}Symbol: {symbol}{RESET}")
                print(f"{BLUE}  ConId: {RESET}{d.contract.conId}")
                print(f"{BLUE}  Exchange: {RESET}{d.contract.exchange}")
                print(f"{BLUE}  Primary Exchange: {RESET}{d.contract.primaryExchange}")
                print(f"{BLUE}  Currency: {RESET}{d.contract.currency}")
                print(f"{BLUE}  Local Symbol: {RESET}{d.contract.localSymbol}")
                print(f"{BLUE}  Trading Class: {RESET}{d.contract.tradingClass}")
                print(f"{MAGENTA}--------------------------------------------{RESET}")

    print()  # Blank line at the end for cleanliness

if __name__ == "__main__":
    main()

