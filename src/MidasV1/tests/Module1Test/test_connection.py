# test_connection.py

"""
========================================================================
# README
#
# Program: test_connection.py
#
# Description:
# This script tests the connectivity between the MidasV1 Trading Bot and the
# Interactive Brokers (IB) Gateway. It attempts to establish a connection,
# request account summaries, and verify that data is being received correctly.
# The script provides colored console outputs to indicate the status of each
# operation, enhancing user feedback and readability.
#
# Features:
# - Checks for the presence of the `ibapi` Python package and prompts installation if missing.
# - Attempts to connect to the IB Gateway using specified host, port, and client ID.
# - Requests account summary data to verify successful communication.
# - Implements colored console outputs using ANSI escape codes:
#     - Green for successful operations.
#     - Red for errors.
#     - Yellow for warnings and informational messages.
#     - Blue and Magenta for decorative separators and headers.
# - Handles graceful shutdown on interrupts and errors.
#
# Usage:
# Run the script from the command line:
#   python test_connection.py
#
# The script will display messages indicating the progress and outcome of each step.
#
# Coded by: kleinpanic 2024
========================================================================
"""

import sys
import time
import threading

# Check for ibapi dependency
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.utils import iswrapper
except ImportError:
    print("┌───────────────────────────────────────────────────┐")
    print("│            IB API Python Not Found!               │")
    print("└───────────────────────────────────────────────────┘")
    print("\nThe 'ibapi' package is not installed. Please install it by running:")
    print("    pip install ibapi")
    sys.exit(1)

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

SEPARATOR = MAGENTA + "────────────────────────────────────────────────────" + RESET

class TestWrapper(EWrapper):
    def __init__(self):
        super().__init__()
        self.nextValidOrderId = None
        self.connected_flag = False
        self.received_account_summary = False

    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Distinguish between known "info" messages and real errors
        # For example, errorCode=2104,2107,2158 are typically info:
        info_codes = {2104, 2107, 2158}
        if errorCode in info_codes:
            # Print these in a different color to indicate they are not severe
            print(f"{YELLOW}[INFO/STATUS] id={reqId}, code={errorCode}, msg={errorString}{RESET}")
        else:
            # True errors
            print(f"{RED}[ERROR] id={reqId}, code={errorCode}, msg={errorString}{RESET}")

    @iswrapper
    def nextValidId(self, orderId: int):
        print(f"{GREEN}[INFO] Next valid order ID: {orderId}{RESET}")
        self.nextValidOrderId = orderId
        self.connected_flag = True

    @iswrapper
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        self.received_account_summary = True
        print(f"{GREEN}[ACCOUNT SUMMARY] ReqId:{reqId}, Account:{account}, {tag} = {value} {currency}{RESET}")

    @iswrapper
    def accountSummaryEnd(self, reqId: int):
        print(f"{GREEN}[ACCOUNT SUMMARY END] ReqId: {reqId}{RESET}")

    @iswrapper
    def managedAccounts(self, accountsList: str):
        print(f"{GREEN}[INFO] Managed accounts: {accountsList}{RESET}")


class TestClient(EClient):
    def __init__(self, wrapper):
        super().__init__(wrapper)


class ConnectionTestApp(TestWrapper, TestClient):
    def __init__(self, host: str, port: int, client_id: int):
        TestWrapper.__init__(self)
        TestClient.__init__(self, self)

        self.host = host
        self.port = port
        self.client_id = client_id

    def connect_and_run(self):
        print(SEPARATOR)
        print(f"{BOLD}{BLUE}    IB Gateway Connection Test{RESET}")
        print(SEPARATOR)
        print(f"{BLUE}Attempting to connect to IB Gateway at {self.host}:{self.port}...{RESET}")

        # Attempt connection with error handling
        try:
            self.connect(self.host, self.port, self.client_id)
        except ConnectionRefusedError:
            print(f"{RED}[ERROR] Connection refused. Is IB Gateway running?{RESET}")
            return

        # Start the EClient message processing thread
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

        # Wait until connected or timeout
        start_time = time.time()
        timeout = 5  # seconds
        while not self.connected_flag and (time.time() - start_time < timeout):
            time.sleep(0.1)

        if not self.connected_flag:
            print(f"{RED}[ERROR] Connection not established within timeout.{RESET}")
            self.disconnect()
            print(f"{YELLOW}[WARN] No connection. Check Gateway settings and try again.{RESET}")
            return

        print(f"{GREEN}[INFO] Connected successfully!{RESET}")
        print(f"{BLUE}Requesting account summary...{RESET}")

        # Request account summary to verify further communication
        req_id = 1
        self.reqAccountSummary(req_id, "All", "NetLiquidation,TotalCashValue,EquityWithLoanValue,BuyingPower")

        # Wait a bit for responses
        time.sleep(5)
        self.cancelAccountSummary(req_id)
        self.disconnect()

        # Check if we received account summary data
        if self.received_account_summary:
            print(f"{GREEN}[INFO] Successfully retrieved account summary data.{RESET}")
        else:
            print(f"{YELLOW}[WARN] Connected but did not receive account summary data. Is the account funded or available?{RESET}")

        print(f"{GREEN}[INFO] Disconnected successfully.{RESET}")
        print(SEPARATOR)
        print(f"{BOLD}{BLUE}    Test Complete{RESET}")
        print(SEPARATOR)


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 4002   # Paper trading port
    client_id = 0 # Choose a unique client ID

    # Check Python version
    if sys.version_info < (3, 6):
        print(f"{RED}[ERROR] Python 3.6+ is required for this script.{RESET}")
        sys.exit(1)

    app = ConnectionTestApp(host, port, client_id)
    app.connect_and_run()

