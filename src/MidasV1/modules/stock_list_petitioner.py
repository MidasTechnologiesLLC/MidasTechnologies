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
# Usage:
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
import os

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

        criteria_message = (
            f"Loaded Scanner Criteria:\n"
            f"  - Search Volume: {search_volume}\n"
            f"  - Net Change: {net_change}\n"
            f"  - Percent Change: {percent_change}"
        )
        self.logger.info(criteria_message)
        print("\033[94m" + criteria_message + "\033[0m")

        # Define the scanner subscription
        subscription = ScannerSubscription()
        subscription.instrument = "STK"
        subscription.locationCode = "STK.US.MAJOR"
        subscription.scanCode = "ALL"
        subscription.aboveVolume = search_volume

        api_call_message = "Initiating scanner subscription with the above criteria..."
        self.logger.info(api_call_message)
        print("\033[92m" + api_call_message + "\033[0m")

        # Optionally, implement retries
        MAX_RETRIES = 2
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.connected_client.reqScannerSubscription(
                    reqId=1001,
                    subscription=subscription,
                    scannerSubscriptionOptions=[],
                    scannerSubscriptionFilterOptions=[]
                )
                self.logger.info(f"Scanner subscription requested successfully on attempt {attempt}.")
                print(f"\033[92mScanner subscription requested successfully on attempt {attempt}.\033[0m")
                break
            except Exception as e:
                self.logger.error(f"Attempt {attempt}: Error in reqScannerSubscription: {e}")
                print(f"\033[91mAttempt {attempt}: Error in reqScannerSubscription: {e}\033[0m")
                if attempt == MAX_RETRIES:
                    return []
                time.sleep(2 ** attempt)

        scanner_timeout = 15
        self.logger.info(f"Waiting for scanner data (timeout in {scanner_timeout} seconds)...")
        print(f"\033[93mWaiting for scanner data (timeout in {scanner_timeout} seconds)...\033[0m")
        scanner_completed = self.scanner_event.wait(timeout=scanner_timeout)
        if not scanner_completed:
            self.logger.error("Scanner subscription timed out.")
            try:
                self.connected_client.cancelScannerSubscription(1001)
                self.logger.error("Scanner subscription canceled due to timeout.")
                print("\033[91mScanner subscription timed out and was canceled.\033[0m")
            except Exception as e:
                self.logger.error(f"Error canceling scanner subscription: {e}")
                print(f"\033[91mError canceling scanner subscription: {e}\033[0m")
            return []

        self.logger.info("Scanner data received. Proceeding to refine the stock list.")
        print("\033[92mScanner data received. Proceeding to refine the stock list.\033[0m")

        self.logger.debug(f"Total scanner data received: {len(self.scanner_data)}")
        print(f"\033[94mTotal scanner data received: {len(self.scanner_data)}\033[0m")

        refined_list = self.refine_stock_list()

        self.cache_refined_list(refined_list)
        self.print_refined_list(refined_list)

        if not refined_list:
            self.logger.error("No stocks meet the specified criteria after refinement.")
            print("\033[91mNo stocks meet the specified criteria after refinement.\033[0m\n")

        # Disconnect the client to prevent further logging
        try:
            self.connected_client.disconnect()
            self.logger.info("Disconnected from IB Gateway after Module 2.")
            print("\033[92mDisconnected from IB Gateway after Module 2.\033[0m")
        except Exception as e:
            self.logger.error(f"Error disconnecting from IB Gateway: {e}")
            print(f"\033[91mError disconnecting from IB Gateway: {e}\033[0m")

        self.logger.info("Module 2: IBJTS List Petitioner completed successfully.")
        print("\033[92mModule 2: IBJTS List Petitioner completed successfully.\033[0m")

        return refined_list

    @iswrapper
    def scannerData(self, reqId: int, rank: int, contractDetails, distance: str,
                    benchmark: str, projection: str, legsStr: str):
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
        self.logger.info(f"Scanner data end received for reqId: {reqId}")
        with self.lock:
            self.scanner_finished = True
            self.scanner_event.set()

    def refine_stock_list(self):
        self.logger.info("Refining the stock list based on criteria...")
        print("\033[93mRefining the stock list based on criteria...\033[0m")

        refined_list = []
        for stock in self.scanner_data:
            symbol = stock['symbol']
            self.logger.debug(f"Processing stock: {symbol}")
            print(f"\033[94mProcessing stock: {symbol}\033[0m")

            # Fetch share price using historical data method
            share_price = self.get_share_price(symbol)
            if share_price is None:
                self.logger.debug(f"Skipping {symbol}: Unable to retrieve share price.")
                print(f"\033[91mSkipping {symbol}: Unable to retrieve share price.\033[0m")
                continue

            if share_price > self.config.getfloat('Module2', 'default_refinement_share_price', fallback=15.0):
                self.logger.debug(f"Excluding {symbol}: Share price ${share_price} exceeds threshold.")
                print(f"\033[91mExcluding {symbol}: Share price ${share_price} exceeds threshold.\033[0m")
                continue

            if not self.has_option_contracts(symbol):
                self.logger.debug(f"Excluding {symbol}: No option contracts available.")
                print(f"\033[91mExcluding {symbol}: No option contracts available.\033[0m")
                continue

            volatility_index = self.get_volatility_index(symbol)
            if volatility_index is None:
                self.logger.debug(f"Skipping {symbol}: Unable to retrieve volatility index.")
                print(f"\033[91mSkipping {symbol}: Unable to retrieve volatility index.\033[0m")
                continue

            if volatility_index > self.config.getfloat('Module2', 'default_volatility_threshold', fallback=30.0):
                self.logger.debug(f"Excluding {symbol}: Volatility index {volatility_index}% exceeds threshold.")
                print(f"\033[91mExcluding {symbol}: Volatility index {volatility_index}% exceeds threshold.\033[0m")
                continue

            refined_list.append({
                'symbol': symbol,
                'share_price': share_price,
                'volatility_index': volatility_index
            })
            self.logger.debug(f"Including {symbol}: Meets all criteria.")
            print(f"\033[92mIncluding {symbol}: Meets all criteria.\033[0m")

        conditional_refinement = self.config.getboolean('Module2', 'conditional_refinement_enabled', fallback=False)
        if conditional_refinement:
            max_list_size = self.config.getint('Module2', 'max_refined_list_size', fallback=100)
            if len(refined_list) > max_list_size:
                refined_list = refined_list[:max_list_size]
                self.logger.info(f"List truncated to {max_list_size} items based on conditional refinement.")
                print(f"\033[93mList truncated to {max_list_size} items based on conditional refinement.\033[0m")

        self.logger.info(f"Refined list contains {len(refined_list)} stocks after applying all filters.")
        print(f"\033[94mRefined list contains {len(refined_list)} stocks after applying all filters.\033[0m")
        return refined_list

    def get_share_price(self, symbol):
        """
        Retrieves the current share price for a given symbol by requesting 1 day of historical data
        and using the close price of the returned bar.

        Returns:
            float: Close price of the latest bar or None if unavailable.
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        price = None
        bars = []
        data_event = threading.Event()

        # Temporary overrides for historical data callbacks
        original_historicalData = self.connected_client.wrapper.historicalData
        original_historicalDataEnd = self.connected_client.wrapper.historicalDataEnd

        def historicalData_override(reqId, bar):
            bars.append(bar)

        def historicalDataEnd_override(reqId, start, end):
            if bars:
                last_bar = bars[-1]
                # Use the close price of the bar as the share price
                nonlocal price
                price = last_bar.close
            data_event.set()

        # Override the callbacks
        self.connected_client.wrapper.historicalData = historicalData_override
        self.connected_client.wrapper.historicalDataEnd = historicalDataEnd_override

        # Request 1 day of historical data
        endDateTime = time.strftime("%Y%m%d %H:%M:%S", time.localtime(time.time())) + " UTC"
        try:
            self.connected_client.reqHistoricalData(
                reqId=5001,
                contract=contract,
                endDateTime=endDateTime,
                durationStr="1 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            self.logger.debug(f"Requested historical data for {symbol}.")
            print(f"\033[94mRequested historical data for {symbol}.\033[0m")
        except Exception as e:
            self.logger.error(f"Error requesting historical data for {symbol}: {e}")
            print(f"\033[91mError requesting historical data for {symbol}: {e}\033[0m")
            # Restore original callbacks
            self.connected_client.wrapper.historicalData = original_historicalData
            self.connected_client.wrapper.historicalDataEnd = original_historicalDataEnd
            return None

        # Wait for data or timeout
        if not data_event.wait(timeout=10):
            self.logger.warning(f"Timeout waiting for historical data for {symbol}.")
            print(f"\033[91mTimeout waiting for historical data for {symbol}.\033[0m")
        else:
            if price is not None:
                self.logger.debug(f"Share price for {symbol}: ${price}")
                print(f"\033[92mShare price for {symbol}: ${price}\033[0m")
            else:
                self.logger.warning(f"No historical data returned for {symbol}.")
                print(f"\033[91mNo historical data returned for {symbol}.\033[0m")

        # Restore the original callbacks
        self.connected_client.wrapper.historicalData = original_historicalData
        self.connected_client.wrapper.historicalDataEnd = original_historicalDataEnd

        return price

    def has_option_contracts(self, symbol):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"

        has_options = False
        option_event = threading.Event()

        original_contractDetails = self.connected_client.wrapper.contractDetails

        def contractDetails_override(reqId, contractDetails):
            nonlocal has_options
            if contractDetails.contract.symbol == symbol:
                has_options = True
                option_event.set()

        self.connected_client.wrapper.contractDetails = contractDetails_override

        try:
            self.connected_client.reqContractDetails(3001, contract)
            self.logger.debug(f"Requested contract details for options of {symbol}.")
            print(f"\033[94mRequested contract details for options of {symbol}.\033[0m")
        except Exception as e:
            self.logger.error(f"Error requesting contract details for {symbol}: {e}")
            print(f"\033[91mError requesting contract details for {symbol}: {e}\033[0m")
            self.connected_client.wrapper.contractDetails = original_contractDetails
            return False

        if not option_event.wait(timeout=5):
            self.logger.warning(f"Timeout while checking options for {symbol}.")
            print(f"\033[91mTimeout while checking options for {symbol}.\033[0m")
        else:
            self.logger.debug(f"Options availability for {symbol}: {has_options}")
            print(f"\033[92mOptions availability for {symbol}: {has_options}\033[0m")

        self.connected_client.wrapper.contractDetails = original_contractDetails
        return has_options

    def get_volatility_index(self, symbol):
        # Placeholder implementation
        mock_volatility = 25.0
        self.logger.debug(f"Volatility index for {symbol}: {mock_volatility}%")
        print(f"\033[94mVolatility index for {symbol}: {mock_volatility}%\033[0m")
        return mock_volatility

    def cache_refined_list(self, refined_list):
        try:
            timestamp = int(time.time())
            cache_path = os.path.join(tempfile.gettempdir(), f"refined_stock_list_{timestamp}.json")
            with open(cache_path, 'w') as tmp_file:
                json.dump(refined_list, tmp_file)
            self.logger.info(f"Refined stock list cached at {cache_path}")
            print(f"\033[92mRefined stock list cached at {cache_path}\033[0m")
        except Exception as e:
            self.logger.error(f"Failed to cache refined stock list: {e}")
            print(f"\033[91mFailed to cache refined stock list: {e}\033[0m")

    def print_refined_list(self, refined_list):
        if not refined_list:
            self.logger.error("No stocks meet the specified criteria after refinement.")
            print("\033[91mNo stocks meet the specified criteria after refinement.\033[0m\n")
            return

        self.logger.info("Refined Stock List:")
        print("\n\033[92mRefined Stock List:\033[0m")
        print("--------------------")
        for stock in refined_list:
            print(f"Symbol: {stock['symbol']}, Share Price: ${stock['share_price']:.2f}, Volatility Index: {stock['volatility_index']}%")
        print("--------------------\n")

