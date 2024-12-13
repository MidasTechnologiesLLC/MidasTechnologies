# modules/stock_list_petitioner.py
"""
========================================================================
# README
#
# Module: stock_list_petitioner.py
#
# Description:
# This module handles the process of requesting and refining a list of stock symbols
# based on predefined criteria. It interacts with the Interactive Brokers (IB) API
# to perform scanner subscriptions, receive market data, and apply filters to
# generate a refined list of stocks suitable for trading strategies.
#
# Features:
# - Initiates scanner subscriptions to retrieve stock data based on criteria such as
#   search volume and net change.
# - Receives and processes scanner data asynchronously using IB API callbacks.
# - Refines the received stock list by applying additional criteria like share price,
#   availability of option contracts, and volatility index.
# - Caches the refined stock list for use by subsequent modules.
# - Provides detailed logging and colored console outputs for better traceability and user feedback.
#
# Usage:
# This module is instantiated and used by `main.py` after initial checks are passed.
#
# Example:
# from modules.stock_list_petitioner import StockListPetitioner
# stock_petitioner = StockListPetitioner(config)
# connected_client = initial_checks.run_all_checks(callback_handlers=[stock_petitioner])
# refined_stock_list = stock_petitioner.run_module()
#
# Coded by: kleinpanic 2024
========================================================================
"""

import logging
import threading
import tempfile
import json
import time
import os  # Added import for os

from ibapi.contract import Contract
from ibapi.scanner import ScannerSubscription
from ibapi.ticktype import TickTypeEnum
from ibapi.utils import iswrapper
from ibapi.tag_value import TagValue

class StockListPetitioner:
    def __init__(self, config):
        self.logger = logging.getLogger('MidasV1.StockListPetitioner')
        self.config = config
        self.connected_client = None  # To be set later

        # Scanner results
        self.scanner_data = []
        self.scanner_finished = False
        self.lock = threading.Lock()

        # Event to signal when scanner data is received
        self.scanner_event = threading.Event()

    def set_client(self, connected_client):
        """
        Sets the connected_client after initial checks.
        """
        self.connected_client = connected_client

    def run_module(self):
        """
        Executes the scanner subscription and refines the stock list.
        """
        if not self.connected_client:
            self.logger.error("Connected client is not set. Cannot proceed with scanner subscription.")
            return []

        self.logger.info("Starting Module 2: IBJTS List Petitioner...")

        # Load scanner criteria from config
        search_volume = self.config.getint('Module2', 'default_search_volume', fallback=10000)
        net_change = self.config.getfloat('Module2', 'default_net_change', fallback=0.0)
        percent_change = self.config.getfloat('Module2', 'default_percent_change', fallback=0.0)

        # Display and log criteria
        criteria_message = (
            f"Loaded Scanner Criteria:\n"
            f"  - Search Volume: {search_volume}\n"
            f"  - Net Change: {net_change}\n"
            f"  - Percent Change: {percent_change}"
        )
        self.logger.info(criteria_message)
        print("\033[94m" + criteria_message + "\033[0m")  # Blue text for criteria

        # Define the scanner subscription
        subscription = ScannerSubscription()
        subscription.instrument = "STK"
        subscription.locationCode = "STK.US.MAJOR"  # Broad location code to include major US stocks
        subscription.scanCode = "ALL"                # Broader scan code to include all stocks
        subscription.aboveVolume = search_volume
        # subscription.netChange = net_change        # Removed for compatibility with "ALL"
        # subscription.percentChange = percent_change  # Removed for compatibility with "ALL"

        # Inform user about the API call
        api_call_message = "Initiating scanner subscription with the above criteria..."
        self.logger.info(api_call_message)
        print("\033[92m" + api_call_message + "\033[0m")  # Green text for API call info

        # Optionally, implement retries
        MAX_RETRIES = 2
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.connected_client.reqScannerSubscription(
                    reqId=1001,
                    subscription=subscription,
                    scannerSubscriptionOptions=[],  # Can be extended based on config
                    scannerSubscriptionFilterOptions=[]  # Can be extended based on config
                )
                self.logger.info(f"Scanner subscription requested successfully on attempt {attempt}.")
                print(f"\033[92mScanner subscription requested successfully on attempt {attempt}.\033[0m")  # Green text
                break
            except Exception as e:
                self.logger.error(f"Attempt {attempt}: Error in reqScannerSubscription: {e}")
                print(f"\033[91mAttempt {attempt}: Error in reqScannerSubscription: {e}\033[0m")  # Red text
                if attempt == MAX_RETRIES:
                    return []
                time.sleep(2 ** attempt)  # Exponential backoff

        # Wait for scanner data or timeout
        scanner_timeout = 15  # seconds
        self.logger.info(f"Waiting for scanner data (timeout in {scanner_timeout} seconds)...")
        print(f"\033[93mWaiting for scanner data (timeout in {scanner_timeout} seconds)...\033[0m")  # Yellow text
        scanner_completed = self.scanner_event.wait(timeout=scanner_timeout)
        if not scanner_completed:
            self.logger.error("Scanner subscription timed out.")
            try:
                self.connected_client.cancelScannerSubscription(1001)
                self.logger.error("Scanner subscription canceled due to timeout.")
                print("\033[91mScanner subscription timed out and was canceled.\033[0m")  # Red text
            except Exception as e:
                self.logger.error(f"Error canceling scanner subscription: {e}")
                print(f"\033[91mError canceling scanner subscription: {e}\033[0m")  # Red text
            return []

        self.logger.info("Scanner data received. Proceeding to refine the stock list.")
        print("\033[92mScanner data received. Proceeding to refine the stock list.\033[0m")  # Green text

        # Log the number of scanner data entries received
        self.logger.debug(f"Total scanner data received: {len(self.scanner_data)}")
        print(f"\033[94mTotal scanner data received: {len(self.scanner_data)}\033[0m")  # Blue text

        for stock in self.scanner_data:
            self.logger.debug(f"Stock: {stock['symbol']}, Volume: {stock['distance']}")
            # Optionally, print detailed scanner data for debugging
            # print(f"\033[94mStock: {stock['symbol']}, Volume: {stock['distance']}\033[0m")  # Blue text

        # Process and refine the scanner data
        refined_list = self.refine_stock_list()

        # Cache the refined list
        self.cache_refined_list(refined_list)

        # Print the refined list to the user
        self.print_refined_list(refined_list)

        if not refined_list:
            self.logger.error("No stocks meet the specified criteria after refinement.")
            print("\033[91mNo stocks meet the specified criteria after refinement.\033[0m\n")  # Red text

        # Disconnect the client to prevent further logging
        try:
            self.connected_client.disconnect()
            self.logger.info("Disconnected from IB Gateway after Module 2.")
            print("\033[92mDisconnected from IB Gateway after Module 2.\033[0m")  # Green text
        except Exception as e:
            self.logger.error(f"Error disconnecting from IB Gateway: {e}")
            print(f"\033[91mError disconnecting from IB Gateway: {e}\033[0m")  # Red text

        self.logger.info("Module 2: IBJTS List Petitioner completed successfully.")
        print("\033[92mModule 2: IBJTS List Petitioner completed successfully.\033[0m")  # Green text

        return refined_list

    @iswrapper
    def scannerData(self, reqId: int, rank: int, contractDetails, distance: str,
                   benchmark: str, projection: str, legsStr: str):
        """
        Receives scanner data.
        """
        with self.lock:
            self.scanner_data.append({
                'rank': rank,
                'symbol': contractDetails.contract.symbol,
                'sectype': contractDetails.contract.secType,
                'exchange': contractDetails.contract.exchange,
                'currency': contractDetails.contract.currency,
                'distance': distance,
                'benchmark': benchmark,
                'projection': projection,
                'legsStr': legsStr
            })
            self.logger.debug(f"Received scanner data: {self.scanner_data[-1]}")

    @iswrapper
    def scannerDataEnd(self, reqId: int):
        """
        Indicates the end of scanner data.
        """
        self.logger.info(f"Scanner data end received for reqId: {reqId}")
        with self.lock:
            self.scanner_finished = True
            self.scanner_event.set()

    def refine_stock_list(self):
        """
        Refines the scanner data based on additional criteria.

        Returns:
            list: Refined list of stocks.
        """
        self.logger.info("Refining the stock list based on criteria...")
        print("\033[93mRefining the stock list based on criteria...\033[0m")  # Yellow text

        refined_list = []
        for stock in self.scanner_data:
            symbol = stock['symbol']
            self.logger.debug(f"Processing stock: {symbol}")
            print(f"\033[94mProcessing stock: {symbol}\033[0m")  # Blue text

            # Fetch additional data for each stock
            share_price = self.get_share_price(symbol)
            if share_price is None:
                self.logger.debug(f"Skipping {symbol}: Unable to retrieve share price.")
                print(f"\033[91mSkipping {symbol}: Unable to retrieve share price.\033[0m")  # Red text
                continue  # Skip if unable to fetch share price

            if share_price > self.config.getfloat('Module2', 'default_refinement_share_price', fallback=15.0):
                self.logger.debug(f"Excluding {symbol}: Share price ${share_price} exceeds threshold.")
                print(f"\033[91mExcluding {symbol}: Share price ${share_price} exceeds threshold.\033[0m")  # Red text
                continue  # Remove stocks above the share price threshold

            if not self.has_option_contracts(symbol):
                self.logger.debug(f"Excluding {symbol}: No option contracts available.")
                print(f"\033[91mExcluding {symbol}: No option contracts available.\033[0m")  # Red text
                continue  # Remove stocks without option contracts

            volatility_index = self.get_volatility_index(symbol)
            if volatility_index is None:
                self.logger.debug(f"Skipping {symbol}: Unable to retrieve volatility index.")
                print(f"\033[91mSkipping {symbol}: Unable to retrieve volatility index.\033[0m")  # Red text
                continue  # Skip if unable to fetch volatility index

            if volatility_index > self.config.getfloat('Module2', 'default_volatility_threshold', fallback=30.0):
                self.logger.debug(f"Excluding {symbol}: Volatility index {volatility_index}% exceeds threshold.")
                print(f"\033[91mExcluding {symbol}: Volatility index {volatility_index}% exceeds threshold.\033[0m")  # Red text
                continue  # Remove stocks above the volatility threshold

            # Append to refined list if all criteria are met
            refined_list.append({
                'symbol': symbol,
                'share_price': share_price,
                'volatility_index': volatility_index
            })
            self.logger.debug(f"Including {symbol}: Meets all criteria.")
            print(f"\033[92mIncluding {symbol}: Meets all criteria.\033[0m")  # Green text

        # Conditional refinement based on config
        conditional_refinement = self.config.getboolean('Module2', 'conditional_refinement_enabled', fallback=False)
        if conditional_refinement:
            max_list_size = self.config.getint('Module2', 'max_refined_list_size', fallback=100)
            if len(refined_list) > max_list_size:
                refined_list = refined_list[:max_list_size]
                self.logger.info(f"List truncated to {max_list_size} items based on conditional refinement.")
                print(f"\033[93mList truncated to {max_list_size} items based on conditional refinement.\033[0m")  # Yellow text

        self.logger.info(f"Refined list contains {len(refined_list)} stocks after applying all filters.")
        print(f"\033[94mRefined list contains {len(refined_list)} stocks after applying all filters.\033[0m")  # Blue text
        return refined_list

    def get_share_price(self, symbol):
        """
        Retrieves the current share price for a given symbol.

        Args:
            symbol (str): Stock symbol.

        Returns:
            float: Current share price or None if unavailable.
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        price = None
        price_event = threading.Event()

        def tickPrice_override(reqId, tickType, price_value, attrib):
            nonlocal price
            if tickType == TickTypeEnum.LAST:
                price = price_value
                price_event.set()

        # Temporarily override the tickPrice callback
        original_tickPrice = self.connected_client.wrapper.tickPrice
        self.connected_client.wrapper.tickPrice = tickPrice_override

        # Request market data
        try:
            self.connected_client.reqMktData(2001, contract, "", False, False, [])
            self.logger.debug(f"Requested market data for {symbol}.")
            print(f"\033[94mRequested market data for {symbol}.\033[0m")  # Blue text
        except Exception as e:
            self.logger.error(f"Error requesting market data for {symbol}: {e}")
            print(f"\033[91mError requesting market data for {symbol}: {e}\033[0m")  # Red text
            self.connected_client.wrapper.tickPrice = original_tickPrice
            return None

        # Wait for the price to be received or timeout
        if not price_event.wait(timeout=5):
            self.logger.warning(f"Timeout while waiting for share price of {symbol}.")
            print(f"\033[91mTimeout while waiting for share price of {symbol}.\033[0m")  # Red text
        else:
            self.logger.debug(f"Share price for {symbol}: ${price}")
            print(f"\033[92mShare price for {symbol}: ${price}\033[0m")  # Green text

        # Restore the original tickPrice callback
        self.connected_client.wrapper.tickPrice = original_tickPrice

        if price is not None:
            return price
        else:
            self.logger.warning(f"Unable to retrieve share price for {symbol}.")
            print(f"\033[91mUnable to retrieve share price for {symbol}.\033[0m")  # Red text
            return None

    def has_option_contracts(self, symbol):
        """
        Checks if option contracts are available for a given symbol.

        Args:
            symbol (str): Stock symbol.

        Returns:
            bool: True if options are available, False otherwise.
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"

        has_options = False
        option_event = threading.Event()

        def contractDetails_override(reqId, contractDetails):
            nonlocal has_options
            if contractDetails.contract.symbol == symbol:
                has_options = True
                option_event.set()

        # Temporarily override the contractDetails callback
        original_contractDetails = self.connected_client.wrapper.contractDetails
        self.connected_client.wrapper.contractDetails = contractDetails_override

        # Request contract details
        try:
            self.connected_client.reqContractDetails(3001, contract)
            self.logger.debug(f"Requested contract details for options of {symbol}.")
            print(f"\033[94mRequested contract details for options of {symbol}.\033[0m")  # Blue text
        except Exception as e:
            self.logger.error(f"Error requesting contract details for {symbol}: {e}")
            print(f"\033[91mError requesting contract details for {symbol}: {e}\033[0m")  # Red text
            self.connected_client.wrapper.contractDetails = original_contractDetails
            return False

        # Wait for the callback or timeout
        if not option_event.wait(timeout=5):
            self.logger.warning(f"Timeout while checking options for {symbol}.")
            print(f"\033[91mTimeout while checking options for {symbol}.\033[0m")  # Red text
        else:
            self.logger.debug(f"Options availability for {symbol}: {has_options}")
            print(f"\033[92mOptions availability for {symbol}: {has_options}\033[0m")  # Green text

        # Restore the original contractDetails callback
        self.connected_client.wrapper.contractDetails = original_contractDetails

        return has_options

    def get_volatility_index(self, symbol):
        """
        Retrieves the volatility index for a given symbol.
        This is a placeholder function. Implement actual volatility retrieval as needed.

        Args:
            symbol (str): Stock symbol.

        Returns:
            float: Volatility index or None if unavailable.
        """
        # Placeholder implementation
        # Replace this with actual implementation, e.g., using an external API or IB's data
        mock_volatility = 25.0  # Example value
        self.logger.debug(f"Volatility index for {symbol}: {mock_volatility}%")
        print(f"\033[94mVolatility index for {symbol}: {mock_volatility}%\033[0m")  # Blue text
        return mock_volatility

    def cache_refined_list(self, refined_list):
        """
        Caches the refined stock list in a temporary file for transfer between modules.

        Args:
            refined_list (list): Refined list of stocks.
        """
        try:
            timestamp = int(time.time())
            cache_path = os.path.join(tempfile.gettempdir(), f"refined_stock_list_{timestamp}.json")
            with open(cache_path, 'w') as tmp_file:
                json.dump(refined_list, tmp_file)
            self.logger.info(f"Refined stock list cached at {cache_path}")
            print(f"\033[92mRefined stock list cached at {cache_path}\033[0m")  # Green text
        except Exception as e:
            self.logger.error(f"Failed to cache refined stock list: {e}")
            print(f"\033[91mFailed to cache refined stock list: {e}\033[0m")  # Red text

    def print_refined_list(self, refined_list):
        """
        Prints the refined stock list to the user.

        Args:
            refined_list (list): Refined list of stocks.
        """
        if not refined_list:
            self.logger.error("No stocks meet the specified criteria after refinement.")
            print("\033[91mNo stocks meet the specified criteria after refinement.\033[0m\n")  # Red text
            return

        self.logger.info("Refined Stock List:")
        print("\n\033[92mRefined Stock List:\033[0m")  # Green text
        print("--------------------")
        for stock in refined_list:
            print(f"Symbol: {stock['symbol']}, Share Price: ${stock['share_price']:.2f}, Volatility Index: {stock['volatility_index']}%")
        print("--------------------\n")

