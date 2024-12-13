# MidasV1

![MidasV1 Logo](https://via.placeholder.com/150) <!-- Replace with actual logo if available -->

## Table of Contents

- [Overview](#overview)
- [General Technical Overview](#general-technical-overview)
- [Workflow & Program Design](#workflow--program-design)
  - [Module 1: Initial Checks](#module-1-initial-checks)
    - [a. Operating System Check](#a-operating-system-check)
    - [b. Dependency Check](#b-dependency-check)
    - [c. Connectivity Check](#c-connectivity-check)
  - [Module 2: IBJTS List Petitioner](#module-2-ibjts-list-petitioner)
    - [a. Scanner](#a-scanner)
    - [b. Refiner](#b-refiner)
  - [Module 3: Stock Information Retrieval](#module-3-stock-information-retrieval)
    - [a. Load](#a-load)
    - [b. Threaded Information Gathering & Choosing Strategy](#b-threaded-information-gathering--choosing-strategy)
    - [c. Strategy Implementation & Market Determination](#c-strategy-implementation--market-determination)
  - [Module 4: Option Chain Trading & Risk Management](#module-4-option-chain-trading--risk-management)
    - [a. Option Chain Data](#a-option-chain-data)
    - [b. Risk Management Stage 1](#b-risk-management-stage-1)
    - [c. Buying and Selling / Risk Management Stage 2](#c-buying-and-selling--risk-management-stage-2)
  - [General Additions](#general-additions)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Logging](#logging)
- [Future Enhancements](#future-enhancements)
- [Disclaimer](#disclaimer)

---

## Overview

**MidasV1** is a sophisticated trading bot developed in Python, designed to interact seamlessly with the Interactive Brokers (IB) Gateway or the IBJTS API and JTS. Leveraging a modular architecture, MidasV1 performs comprehensive system checks, retrieves and refines stock data, executes trading strategies based on real-time market analysis, and manages risks effectively. 

This README provides an in-depth overview of MidasV1's architecture, functionalities, and setup instructions.

---

## General Technical Overview

- **Programming Language:** Python
- **Dependencies:**
  - `ibapi`: Interactive Brokers API for Python
  - `psutil`: For system resource monitoring
- **Requirements:**
  - **IBJTS API & JTS:** Ensure that the IBJTS API and JTS are running.
  - **IB Gateway:** Alternatively, IB Gateway can be used for connectivity.
- **Architecture:** Highly modular, facilitating scalability and maintainability.

---

## Workflow & Program Design

MidasV1 is structured into multiple modules, each responsible for distinct functionalities, ensuring a clear separation of concerns and streamlined operations.

### Module 1: Initial Checks

#### a. Operating System Check

- **Purpose:** Determine the operating system of the device running MidasV1.
- **Supported OS:** Linux
- **Unsupported OS:** Windows, MacOS, BSD, illumos, etc.
- **Behavior:**
  - **Linux:** Continues with execution, providing a green success message.
  - **Unsupported OSes:** Displays red error messages and yellow warnings indicating future support considerations, then gracefully exits.

#### b. Dependency Check

- **Purpose:** Ensure all necessary Python packages are installed.
- **Mechanism:**
  - **Source:** Reads dependencies from `requirements.txt` located in the parent directory of the `modules` folder.
  - **Process:**
    - Parses `requirements.txt` to extract package names.
    - Checks if each package is installed using `pkg_resources`.
    - **Missing Dependencies:** Informs the user with red text and provides the exact `pip` command to install them.
    - **All Dependencies Present:** Confirms with a green success message.
  
#### c. Connectivity Check

- **Purpose:** Verify a secure and functional connection with the IB Gateway.
- **Configuration:**
  - **Source:** Retrieves `host`, `port`, and `client_id` from `config/config.config` with default fallbacks (`127.0.0.1`, `4002`, `0` respectively).
- **Behavior:**
  - Attempts to establish a connection to the IB Gateway.
  - **Success:** Displays green messages confirming connection and successful retrieval of account summaries.
  - **Failure:** Shows red error messages and yellow warnings, then exits gracefully.
- **Enhancements:**
  - Utilizes colored console outputs for clear and human-readable reporting.
  - Addresses the unused `advancedOrderRejectJson` parameter to eliminate warnings.

### Module 2: IBJTS List Petitioner

#### a. Scanner

- **Purpose:** Retrieve a list of stocks that meet predefined criteria.
- **Configuration:** Loads criteria such as search volume, net change, and percent change from `config/config.config`.
- **Process:**
  - Initiates a scanner subscription via the IB API using the loaded criteria.
  - Requests a list of stocks that satisfy the specified metrics.
  - Caches the retrieved list temporarily for further processing.

#### b. Refiner

- **Purpose:** Further refine the scanned stock list based on additional criteria.
- **Criteria:**
  1. **Share Price:** Exclude stocks with share prices exceeding a threshold defined in the config.
  2. **Option Contracts:** Remove stocks without available option contracts.
  3. **Volatility Index:** Exclude stocks with a volatility index above the configured threshold.
  4. **Conditional Truncation:** If enabled, truncate the list to a maximum size specified in the config.
- **Behavior:**
  - Applies each refinement step sequentially.
  - Provides colored console outputs indicating inclusion or exclusion of stocks.
  - Caches the refined list for transfer to subsequent modules.

### Module 3: Stock Information Retrieval

#### a. Load

- **Purpose:** Load the refined stock list from Module 2.
- **Behavior:** Ensures the availability of the refined list and prepares it for data retrieval.

#### b. Threaded Information Gathering & Choosing Strategy

- **Purpose:** Gather real-time market data for each stock and determine the optimal trading strategy.
- **Process:**
  - **Threading:** Spawns individual threads for each stock to fetch data asynchronously.
  - **Data Retrieved:** Datetime, high, low, close, and volume at specified trading intervals.
  - **Data Storage:** Saves data in JSON files named `{stock_name}.{current_date}.json` within the `data/` directory.
  - **Strategy Counter:** Maintains a counter based on incoming data to determine trend indicators.

#### c. Strategy Implementation & Market Determination

- **Purpose:** Analyze collected data to identify bullish or bearish trends.
- **Indicators:**
  - **RSI (Relative Strength Index)**
  - **MACD (Moving Average Convergence Divergence)**
  - **ADX (Average Directional Index)**
  - **EMA (Exponential Moving Average)**
- **Behavior:**
  - Calculates or retrieves indicator values.
  - Assigns weights to each indicator based on predefined thresholds.
  - Determines overall market sentiment (bullish/bearish) for each stock.
  - Based on internal boolean flags, decides whether to process the entire list or isolate the most bullish and bearish stocks for further actions.

### Module 4: Option Chain Trading & Risk Management

#### a. Option Chain Data

- **Purpose:** Retrieve and analyze option chain data for selected stocks.
- **Process:**
  - **Data Retrieval:** Fetches option contracts closest to the current share price.
  - **Filtering:**
    - **Bearish Stocks:** Isolates contracts with strike prices slightly above the share price.
    - **Bullish Stocks:** Isolates contracts with strike prices slightly below the share price.
  - **Behavior:** Ensures that contracts are selected based on proximity to the current market price and other configurable parameters.

#### b. Risk Management Stage 1

- **Purpose:** Assess the acceptability of risk before executing trades.
- **Process:**
  - Retrieves user account balance information.
  - Determines if the cost of option contracts is within the acceptable risk percentage defined in the config.
  - **Outcome:** Only proceeds with contracts that meet the risk criteria.

#### c. Buying and Selling / Risk Management Stage 2

- **Purpose:** Execute trades and manage ongoing risk.
- **Process:**
  - **Trade Execution:** Buys option contracts that passed risk assessments.
  - **Stop-Loss Orders:** Sets up stop-loss contracts based on configurable loss thresholds.
  - **Continuous Monitoring:** Gathers real-time data to implement selling strategies, ensuring optimal trade exits.

### General Additions

- **Command-Line Flags:**
  - `--no-checks`: Runs the program without prompting for user confirmation after initial checks.
  - `--skip-checks`: Skips specific initial checks (primarily dependency checks).
  - `--verbose`: Enables verbose and colorful output to the console.
  - `--version`: Prints the program version and exits.
- **Logging & Console Outputs:**
  - Implements both logging to files and colored console outputs.
  - Controlled via the `--verbose` flag to manage verbosity levels.
- **Graceful Shutdowns:**
  - Handles interrupt signals (e.g., Ctrl+C) to ensure connections are closed properly.
- **Extensibility:**
  - Designed to determine the number of threads based on system resources for optimal performance.

---

## File Structure

```
MidasV1/
├── README.md
├── requirements.txt
├── config/
│   └── config.config
├── main.py
├── modules/
│   ├── initial_checks.py
│   └── stock_list_petitioner.py
├── tests/
│   ├── test_data_retriever.py
│   ├── test_stock_retriever.py
│   └── test_connection.py
├── logs/
│   └── MidasV1.log
└── data/
    └── {stock_name}.{current_date}.json
```

- **README.md:** This documentation file.
- **requirements.txt:** Lists all Python dependencies required by MidasV1.
- **config/config.config:** Configuration file containing all necessary parameters and thresholds.
- **main.py:** The primary script that orchestrates the application's flow.
- **modules/:** Contains all modular components of MidasV1.
  - **initial_checks.py:** Performs system and environment checks.
  - **stock_list_petitioner.py:** Retrieves and refines stock lists based on criteria.
- **tests/:** Contains test scripts for various modules.
  - **test_data_retriever.py:** Tests data retrieval functionalities.
  - **test_stock_retriever.py:** Tests stock retrieval and filtering.
  - **test_connection.py:** Tests connectivity with IB Gateway.
- **logs/MidasV1.log:** Logs detailed execution information.
- **data/:** Stores JSON files with raw market data for each stock.

---

## Installation

### Prerequisites

- **Python 3.6 or higher:** Ensure Python is installed on your system. You can verify the installation by running:
  ```bash
  python --version
  ```
  
- **Interactive Brokers (IB) Account:** Required to access the IB Gateway or IBJTS API.

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/MidasV1.git
   cd MidasV1
   ```

2. **Set Up Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download IB API:**
   - **From Interactive Brokers:**
     - Visit the [Interactive Brokers API](https://www.interactivebrokers.com/en/index.php?f=5041) download page.
     - Download and install the latest version of the IB API.
   - **IB Gateway:**
     - Alternatively, install the [IB Gateway](https://www.interactivebrokers.com/en/index.php?f=16457) for a lightweight connection.

5. **Configure IB Gateway:**
   - Launch IB Gateway and log in using your IB credentials.
   - Ensure that the API settings allow connections from your machine:
     - **Enable API:** Navigate to `Configure` > `Settings` > `API` > `Settings`.
     - **Trusted IPs:** Add `127.0.0.1` or your specific IP address.
     - **Port:** Ensure it matches the `port` specified in `config/config.config` (default is `4002`).

6. **Verify Configuration:**
   - Ensure `config/config.config` is properly set with your desired parameters.
   - Example configuration is provided below.

---

## Configuration

All configurable parameters are stored in `config/config.config`. Below is an example of the configuration file:

```ini
[General]
version = 1.0.0
# Future general configurations can be added here

[Connectivity]
host = 127.0.0.1
port = 4002
client_id = 0
# Add more connectivity parameters as needed

[SystemResources]
# Placeholder for system resource related configurations
# Example:
# max_cpu_threads = 8
# min_available_ram_gb = 4

[Logging]
level = INFO
# Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
# Set to DEBUG for verbose logging, INFO for standard logging, etc.

[Module2]
default_search_volume = 1000000
default_net_change = 0.50
default_percent_change = 2.0
default_refinement_share_price = 15.0
default_volatility_threshold = 30.0
conditional_refinement_enabled = True
max_refined_list_size = 100
```

### Configuration Sections

- **[General]:** General settings, including the version of MidasV1.
  
- **[Connectivity]:** 
  - **host:** IP address of the IB Gateway (default `127.0.0.1`).
  - **port:** Port number for the API connection (default `4002` for IB Gateway Simulated Trading).
  - **client_id:** Unique client ID for the API connection.

- **[SystemResources]:** 
  - Placeholder for future configurations related to system resources.
  - Example parameters for thread management can be added here.

- **[Logging]:** 
  - **level:** Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

- **[Module2]:** 
  - **default_search_volume:** Minimum trading volume for stock selection.
  - **default_net_change:** Minimum net change in stock price.
  - **default_percent_change:** Minimum percentage change in stock price.
  - **default_refinement_share_price:** Maximum share price threshold.
  - **default_volatility_threshold:** Maximum acceptable volatility index.
  - **conditional_refinement_enabled:** Boolean to enable or disable list truncation.
  - **max_refined_list_size:** Maximum number of stocks in the refined list.

---

## Usage

### Running the Application

Navigate to the project root directory and execute `main.py` with desired flags:

```bash
python main.py [--no-checks] [--skip-checks] [--verbose] [--version]
```

### Available Flags

- `--no-checks`: Run the program without prompting for user confirmation after initial checks.
- `--skip-checks`: Skip specific initial checks (primarily dependency checks).
- `--verbose`: Enable verbose and colorful output to the console.
- `--version`: Print the program version and exit.

### Example Commands

1. **Standard Execution:**
   ```bash
   python main.py
   ```

2. **Verbose Mode:**
   ```bash
   python main.py --verbose
   ```

3. **Skip Dependency Checks:**
   ```bash
   python main.py --skip-checks
   ```

4. **Run Without User Confirmation:**
   ```bash
   python main.py --no-checks
   ```

5. **Display Version:**
   ```bash
   python main.py --version
   ```

### Testing Modules

MidasV1 includes several test scripts located in the `tests/` directory to verify the functionality of individual modules.

1. **Test Stock Retriever:**
   ```bash
   python tests/test_stock_retriever.py
   ```

2. **Test Connection:**
   ```bash
   python tests/test_connection.py
   ```

3. **Test Data Retrieval:**
   ```bash
   python tests/test_data_retriever.py
   ```

*Note: Ensure that the IB Gateway or IBJTS API is running before executing test scripts.*

---

## Logging

MidasV1 utilizes both file-based logging and console outputs to track its operations.

- **Log File:** `logs/MidasV1.log`
  - **Location:** Stored in the `logs/` directory within the project root.
  - **Content:** Detailed logs including debug information, errors, and informational messages.
  - **Configuration:** Controlled via the `[Logging]` section in `config/config.config`.

- **Console Outputs:**
  - **Color-Coded Messages:** Enhances readability with green for successes, red for errors, yellow for warnings, and blue/magenta for informational and decorative messages.
  - **Verbosity:** Managed via the `--verbose` flag and the logging level set in the configuration file.

*Ensure that the `logs/` directory exists or is created before running the application to prevent logging errors.*

---

## Future Enhancements

MidasV1 is designed with scalability in mind, allowing for future feature additions and optimizations.

1. **Operating System Support:**
   - Extend support to Windows, MacOS, BSD, illumos, etc., with specific handling mechanisms.

2. **Advanced Dependency Management:**
   - Implement dynamic dependency resolution and version management.

3. **Enhanced Strategy Module:**
   - Develop more sophisticated trading strategies based on additional market indicators.
   - Incorporate machine learning algorithms for predictive analysis.

4. **Risk Management Enhancements:**
   - Implement multi-stage risk assessments.
   - Integrate portfolio diversification strategies.

5. **Performance Optimization:**
   - Utilize system resource checks to dynamically allocate threads for optimal performance.
   - Implement rate limiting and efficient data handling mechanisms.

6. **User Interface:**
   - Develop a graphical user interface (GUI) for easier interaction and monitoring.
   - Provide real-time dashboards for tracking trades and system status.

7. **Extensive Testing:**
   - Expand test coverage to include integration and stress tests.
   - Implement continuous integration (CI) pipelines for automated testing.

8. **Documentation & Support:**
   - Enhance documentation with tutorials and usage guides.
   - Provide support mechanisms for troubleshooting and user assistance.

---

## Disclaimer

**MidasV1** is proprietary software developed for private use. Unauthorized distribution, replication, or modification is strictly prohibited. The author assumes no responsibility for any misuse or damages resulting from the use of this software. Users are advised to thoroughly test the application in a controlled environment (e.g., paper trading) before deploying it in live trading scenarios.

---

## Additional Information

- **Contact:** For inquiries or support, please contact [kleinpanic@gmail.com](mailto:kleinpanic@gmail.com).
- **License:** All rights reserved. No part of this software may be reproduced or transmitted in any form or by any means without the prior written permission of the author.

---

