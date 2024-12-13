# main.py
"""
========================================================================
# README
#
# Program: main.py
#
# Description:
# This script serves as the entry point for the MidasV1 Trading Bot.
# It performs initial system checks, handles configuration loading, sets up logging,
# initializes modules, and manages the overall flow of the application.
#
# Features:
# - Parses command-line arguments for customization.
# - Loads configuration settings from a config file.
# - Sets up colored logging for enhanced visibility.
# - Handles graceful shutdown on interrupt signals (e.g., Ctrl+C).
# - Initializes and runs initial checks and the Stock List Petitioner module.
# - Provides placeholders for integrating additional modules in the future.
#
# Usage:
# Run the script from the command line with optional arguments:
#   --no-checks      : Run the program without prompting for user confirmation after initial checks.
#   --skip-checks    : Skip specific initial checks (primarily dependency checks).
#   --verbose        : Enable verbose and colorful output to the console.
#   --version        : Print the program version and exit.
#
# Example:
#   python main.py --verbose
#   python main.py --version
#   python main.py --verbose --no-checks
#   python main.py --verbose --skip-checks
#
# Coded by: kleinpanic 2024
========================================================================
"""

import argparse
import logging
import sys
import os
import configparser
import signal
import threading

from modules.initial_checks import InitialChecks
from modules.stock_list_petitioner import StockListPetitioner  # Import Module 2

class ColoredFormatter(logging.Formatter):
    # ANSI escape codes for colors
    COLOR_CODES = {
        'DEBUG': "\033[94m",    # Blue
        'INFO': "\033[92m",     # Green
        'WARNING': "\033[93m",  # Yellow
        'ERROR': "\033[91m",    # Red
        'CRITICAL': "\033[95m", # Magenta
    }
    RESET_CODE = "\033[0m"

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelname, self.RESET_CODE)
        message = super().format(record)
        if record.levelname in self.COLOR_CODES:
            message = f"{color}{message}{self.RESET_CODE}"
        return message

def setup_logging(verbose=False, log_level='INFO'):
    """
    Configures logging for the application.

    Args:
        verbose (bool): If True, set logging level to DEBUG and log to console with colors.
        log_level (str): Specific logging level from config.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = os.path.join('logs', 'MidasV1.log')
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Determine logging level
    level = getattr(logging, log_level.upper(), logging.INFO)
    if verbose:
        level = logging.DEBUG

    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler (uncolored, all logs)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Log all levels to file
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    if verbose:
        # Console handler (colored, INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

def load_config(config_path='config/config.config'):
    """
    Loads the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        configparser.ConfigParser: Parsed configuration.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        print(f"Configuration file not found at {config_path}. Exiting.")
        sys.exit(1)
    config.read(config_path)
    return config

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='MidasV1 Trading Bot')
    parser.add_argument('--no-checks', action='store_true', help='Run the program without prompting for user confirmation after initial checks')
    parser.add_argument('--skip-checks', action='store_true', help='Skip specific initial checks (primarily dependency checks)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose and colorful output to the console')
    parser.add_argument('--version', action='store_true', help='Print the program version and exit')
    return parser.parse_args()

# Initialize connected_client as global
connected_client = None

def signal_handler(sig, frame):
    """
    Handles incoming signals for graceful shutdown.

    Args:
        sig (int): Signal number.
        frame: Current stack frame.
    """
    logger = logging.getLogger('MidasV1.Main')
    logger.error("Interrupt received. Shutting down gracefully...")
    global connected_client
    if connected_client:
        connected_client.disconnect()
        logger.info("Disconnected from IB Gateway.")
    sys.exit(0)

def main():
    global connected_client  # Declare as global to modify the global variable

    # Register the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config()
    
    # Setup logging based on flags and config
    log_level = config.get('Logging', 'level', fallback='INFO')
    setup_logging(verbose=args.verbose, log_level=log_level)
    logger = logging.getLogger('MidasV1.Main')
    
    # Suppress ibapi internal logs from propagating to root logger
    ibapi_logger = logging.getLogger('ibapi')
    ibapi_logger.setLevel(logging.WARNING)  # Suppress DEBUG and INFO
    ibapi_logger.propagate = False

    # Handle --version flag
    if args.version:
        version = config.get('General', 'version', fallback='1.0.0')
        print(f"MidasV1 Version: {version}")
        sys.exit(0)
    
    logger.info("Starting MidasV1 Trading Bot...")
    
    # Initialize and run initial checks if not skipping all checks
    if not args.skip_checks:
        try:
            initial_checks = InitialChecks(config, verbose=args.verbose)
            stock_petitioner = StockListPetitioner(config)  # Instantiate with only config
            connected_client = initial_checks.run_all_checks(skip_checks=args.skip_checks, callback_handlers=[stock_petitioner])  # Pass as callback handler
            stock_petitioner.set_client(connected_client)  # Set the connected client
            refined_stock_list = stock_petitioner.run_module()
            logger.info(f"Refined Stock List: {refined_stock_list}")  # Log to prevent unused warning
        except SystemExit as e:
            logger.error("Initial checks failed. Exiting program.")
            sys.exit(e.code)
        except Exception as e:
            logger.exception("An unexpected error occurred during initial checks or Module 2.")
            if connected_client:
                connected_client.disconnect()
                logger.info("Disconnected from IB Gateway.")
            sys.exit(1)
    else:
        logger.warning("Skipping specific initial checks as per the '--skip-checks' flag.")
        refined_stock_list = []
    
    # Prompt the user to confirm before proceeding to the next module
    if not args.no_checks:
        logger.info("Initial checks and Module 2 completed. Please review the logs and ensure everything is correct.")
        try:
            while True:
                user_input = input("Do you want to proceed to the next module? (y/n): ").strip().lower()
                if user_input == 'y':
                    logger.info("User chose to proceed.")
                    break
                elif user_input == 'n':
                    logger.info("User chose to exit the program.")
                    if connected_client:
                        connected_client.disconnect()
                        logger.info("Disconnected from IB Gateway.")
                    sys.exit(0)
                else:
                    print("Please enter 'y' or 'n'.")
        except KeyboardInterrupt:
            logger.error("Interrupt received during user prompt. Shutting down gracefully...")
            if connected_client:
                connected_client.disconnect()
                logger.info("Disconnected from IB Gateway.")
            sys.exit(0)
    
    else:
        logger.info("Proceeding to the next module without user confirmation as per the '--no-checks' flag.")
    
    # Placeholder for initializing and running other modules (e.g., Module 3)
    # Example:
    # from modules.module3 import Module3
    # module3 = Module3(config, connected_client, refined_stock_list)
    # module3.run()
    
    logger.info("MidasV1 Trading Bot is now running.")
    
    # Placeholder for main loop or orchestration logic
    try:
        while True:
            # Implement the main functionality here
            # For demonstration, we'll just sleep
            threading.Event().wait(1)
    except KeyboardInterrupt:
        logger.error("Interrupt received. Shutting down gracefully...")
        if connected_client:
            connected_client.disconnect()
            logger.info("Disconnected from IB Gateway.")
        sys.exit(0)

if __name__ == '__main__':
    main()

