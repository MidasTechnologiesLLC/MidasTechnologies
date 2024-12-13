# modules/initial_checks.py
"""
========================================================================
# README
#
# Module: initial_checks.py
#
# Description:
# This module performs a series of initial system and environment checks
# required before the MidasV1 Trading Bot can operate effectively.
# It verifies the operating system, checks for necessary dependencies,
# assesses system resources, and ensures connectivity with the Interactive Brokers (IB) Gateway.
#
# Features:
# - Checks if the operating system is supported (currently Linux).
# - Verifies that required Python packages (`ibapi`, `psutil`) are installed.
# - Logs detailed system resource information including CPU cores, clock speed,
#   load averages, CPU threads, and RAM statistics.
# - Tests connectivity with the IB Gateway by attempting to establish a session
#   and retrieve account summaries.
# - Integrates with other modules via callback handlers to facilitate data exchange.
#
# Usage:
# This module is primarily used by `main.py` during the startup phase.
#
# Example:
# from modules.initial_checks import InitialChecks
# config = load_config()
# initial_checks = InitialChecks(config, verbose=True)
# connected_client = initial_checks.run_all_checks(callback_handlers=[stock_petitioner])
#
# Coded by: kleinpanic 2024
========================================================================
"""

import platform
import sys
import logging
import os
import psutil
import threading
import time

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.utils import iswrapper

from ibapi.contract import Contract
from ibapi.account_summary_tags import AccountSummaryTags  # Ensure correct import

from ibapi.tag_value import TagValue

SEPARATOR = "────────────────────────────────────────────────────"

class InitialChecks:
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.logger = logging.getLogger('MidasV1.InitialChecks')

    def check_os(self):
        """
        Determines the operating system and verifies if it's supported.
        Currently supports only Linux.
        """
        self.logger.info("Checking Operating System...")
        os_type = platform.system()
        if os_type != 'Linux':
            message = f"Unsupported Operating System: {os_type}"
            self.logger.error(message)
            self.logger.warning("Future support for other operating systems is being added.")
            sys.exit(1)
        success_message = f"Operating System {os_type} is supported."
        self.logger.info(success_message)

    def check_dependencies(self):
        """
        Ensures that all necessary dependencies are installed.
        """
        self.logger.info("Checking Dependencies...")
        # Check if 'ibapi' and 'psutil' are installed
        dependencies = ['ibapi', 'psutil']
        missing = []
        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        if missing:
            error_message = f"Missing Dependencies: {', '.join(missing)}"
            self.logger.error(error_message)
            self.logger.warning("Please install the missing dependencies and try again.")
            sys.exit(1)
        success_message = "All dependencies are satisfied."
        self.logger.info(success_message)

    def check_connectivity(self, callback_handlers=[]):
        """
        Verifies a secure connection with the IB Gateway.

        Args:
            callback_handlers (list): List of modules to receive callbacks.
        """
        self.logger.info("Checking Connectivity with IB Gateway...")

        host = self.config.get('Connectivity', 'host', fallback='127.0.0.1')
        port = self.config.getint('Connectivity', 'port', fallback=4002)
        client_id = self.config.getint('Connectivity', 'client_id', fallback=0)

        # Define wrapper and client for connection test
        class TestWrapper(EWrapper):
            def __init__(self, logger, callback_handlers):
                super().__init__()
                self.nextValidOrderId = None
                self.connected_flag = False
                self.received_account_summary = False
                self.logger = logger
                self.callback_handlers = callback_handlers

            @iswrapper
            def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
                info_codes = {2104, 2106, 2107, 2158}  # Include code 2106 based on your logs
                if errorCode in info_codes:
                    self.logger.info(f"[INFO/STATUS] id={reqId}, code={errorCode}, msg={errorString}")
                else:
                    self.logger.error(f"[ERROR] id={reqId}, code={errorCode}, msg={errorString}")

            @iswrapper
            def nextValidId(self, orderId: int):
                self.logger.info(f"[INFO] Next valid order ID: {orderId}")
                self.nextValidOrderId = orderId
                self.connected_flag = True

            @iswrapper
            def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
                self.received_account_summary = True
                self.logger.info(f"[ACCOUNT SUMMARY] ReqId:{reqId}, Account:{account}, {tag} = {value} {currency}")

            @iswrapper
            def accountSummaryEnd(self, reqId: int):
                self.logger.info(f"[ACCOUNT SUMMARY END] ReqId: {reqId}")

            @iswrapper
            def managedAccounts(self, accountsList: str):
                self.logger.info(f"[INFO] Managed accounts: {accountsList}")

            @iswrapper
            def connectionClosed(self):
                self.logger.error("[ERROR] Connection to IB Gateway was closed unexpectedly!")
                # Notify callback handlers if needed

            @iswrapper
            def scannerData(self, reqId: int, rank: int, contractDetails: Contract, distance: str,
                           benchmark: str, projection: str, legsStr: str):
                # Dispatch to callback handlers
                for handler in self.callback_handlers:
                    handler.scannerData(reqId, rank, contractDetails, distance, benchmark, projection, legsStr)

            @iswrapper
            def scannerDataEnd(self, reqId: int):
                self.logger.info(f"Scanner data end received for reqId: {reqId}")
                for handler in self.callback_handlers:
                    handler.scannerDataEnd(reqId)

        class TestClient(EClient):
            def __init__(self, wrapper):
                super().__init__(wrapper)

        class ConnectionTestApp(TestWrapper, TestClient):
            def __init__(self, host: str, port: int, client_id: int, logger, callback_handlers=[]):
                TestWrapper.__init__(self, logger, callback_handlers)
                TestClient.__init__(self, self)
                self.host = host
                self.port = port
                self.client_id = client_id

            def connect_and_run(self):
                self.logger.info(SEPARATOR)
                self.logger.info("    IB Gateway Connection Test")
                self.logger.info(SEPARATOR)
                connection_message = f"Attempting to connect to IB Gateway at {self.host}:{self.port}..."
                self.logger.info(connection_message)

                try:
                    self.connect(self.host, self.port, self.client_id)
                except ConnectionRefusedError:
                    error_message = "[ERROR] Connection refused. Is IB Gateway running?"
                    self.logger.error(error_message)
                    sys.exit(1)

                # Start the EClient message processing thread
                thread = threading.Thread(target=self.run, daemon=True)
                thread.start()

                # Wait until connected or timeout
                start_time = time.time()
                timeout = 5  # seconds
                while not self.connected_flag and (time.time() - start_time < timeout):
                    time.sleep(0.1)

                if not self.connected_flag:
                    error_message = "[ERROR] Connection not established within timeout."
                    self.logger.error(error_message)
                    warning_message = "[WARN] No connection. Check Gateway settings and try again."
                    self.logger.warning(warning_message)
                    self.disconnect()
                    sys.exit(1)

                success_message = "[INFO] Connected successfully!"
                self.logger.info(success_message)

                request_message = "Requesting account summary..."
                self.logger.info(request_message)

                # Request account summary to verify further communication
                req_id = 1
                self.reqAccountSummary(req_id, "All", "NetLiquidation,TotalCashValue,EquityWithLoanValue,BuyingPower")

                # Wait a bit for responses
                time.sleep(5)
                self.cancelAccountSummary(req_id)

                # Check if we received account summary data
                if self.received_account_summary:
                    success_summary = "[INFO] Successfully retrieved account summary data."
                    self.logger.info(success_summary)
                else:
                    warning_summary = "[WARN] Connected but did not receive account summary data. Is the account funded or available?"
                    self.logger.warning(warning_summary)

                self.logger.info("IB Gateway is connected and ready for upcoming modules.")
                self.logger.info(SEPARATOR)
                self.logger.info("    Test Complete")
                self.logger.info(SEPARATOR)

        # Initialize and run the connection test
        app = ConnectionTestApp(host, port, client_id, self.logger, callback_handlers)
        app.connect_and_run()

        # Return the app to keep the connection open
        return app

    def check_system_resources(self):
        """
        Logs system resource information such as CPU cores, clock speed, load averages, CPU threads, and detailed RAM information.
        The system information is colored gold for better visibility.
        """
        self.logger.info("Checking System Resources...")

        # Gather system information
        cpu_cores = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_load_avg = psutil.getloadavg()
        cpu_threads = psutil.cpu_count(logical=True)
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()

        total_ram_gb = ram.total / (1024 ** 3)
        used_ram_gb = ram.used / (1024 ** 3)
        available_ram_gb = ram.available / (1024 ** 3)
        ram_percent = ram.percent

        swap_total_gb = swap.total / (1024 ** 3)
        swap_used_gb = swap.used / (1024 ** 3)
        swap_percent = swap.percent

        # Construct the resource information string
        resource_info = (
            f"CPU Cores: {cpu_cores}\n"
            f"CPU Clock Speed: {cpu_freq.current:.2f} MHz\n"
            f"CPU Load Average (1m, 5m, 15m): {cpu_load_avg}\n"
            f"CPU Threads: {cpu_threads}\n"
            f"Total RAM: {total_ram_gb:.2f} GB\n"
            f"Used RAM: {used_ram_gb:.2f} GB ({ram_percent}%)\n"
            f"Available RAM: {available_ram_gb:.2f} GB\n"
            f"Total Swap: {swap_total_gb:.2f} GB\n"
            f"Used Swap: {swap_used_gb:.2f} GB ({swap_percent}%)"
        )

        # ANSI escape code for gold (approximated by yellow)
        gold_color = "\033[93m"  # Bright Yellow as gold approximation
        reset_color = "\033[0m"

        # Combine the newline and colored resource information
        colored_resource_info = f"\n{gold_color}{resource_info}{reset_color}"

        # Log the colored resource information
        self.logger.info(colored_resource_info)

    def run_all_checks(self, skip_checks=False, callback_handlers=[]):
        """
        Executes all initial checks in the required sequence.

        Args:
            skip_checks (bool): If True, skips specific checks like dependency checks.
            callback_handlers (list): List of modules to receive callbacks.

        Returns:
            connected_client: The connected ibapi client to be used by other modules.
        """
        self.check_os()
        if not skip_checks:
            self.check_dependencies()
        else:
            warning_message = "Skipping dependency checks as per the '--skip-checks' flag."
            self.logger.warning(warning_message)
        self.check_system_resources()
        connected_client = self.check_connectivity(callback_handlers)
        success_message = "All initial checks passed successfully."
        self.logger.info(success_message)
        return connected_client

