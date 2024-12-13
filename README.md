# Midas Technologies LLC

## Overview

Welcome to **Midas Technologies LLC**, an innovative company focused on developing sophisticated algorithmic trading solutions, primarily aimed at the financial markets. Our goal is to deliver above-market returns using a robust, data-driven approach powered by advanced technology, natural language processing (NLP), machine learning, and state-of-the-art trading algorithms.

## Mission Statement

Midas Technologies aims to build and manage a diversified portfolio of algorithmic trading strategies that maximize returns while managing risk effectively. With a commitment to continual improvement and innovation, we strive to provide top-tier trading systems capable of navigating volatile markets and delivering consistent profitability.

## Business Model

Our core product is a modular algorithmic trading platform. The current focus is **MidasV1**, a trading bot for contracts and options, with functionality spanning real-time data collection, market analysis, and automated execution.

### **Current Project: MidasV1**

#### Purpose
MidasV1 automates trading decisions by combining advanced market analysis techniques, sentiment scoring, and option chain evaluation to ensure optimal trading performance.

#### Workflow / Program Design
1. **Module 1: Initial System Checks**
   - **Operating System Check**: Ensures compatibility with the host system (default: Linux).
   - **Dependency Check**: Verifies that all required libraries and tools are installed.
   - **Connectivity Check**: Confirms secure integration with IBJTS or IB Gateway.

2. **Module 2: IBJTS List Petitioner**
   - Scans and refines a list of stocks meeting initial volume, change, and percent change criteria.
   - Filters stocks based on share price, options availability, volatility, and configurable thresholds.

3. **Module 3: Stock Information Retrieval**
   - Gathers historical and intraday trading data (datetime, high, low, close, volume).
   - Implements a strategy counter to determine the best indicators (e.g., RSI, MACD, ADX) for market analysis.

4. **Module 4: Option Chain Trading and Risk Management**
   - Evaluates option chain data for selected bullish and bearish stocks.
   - Executes trades with dynamic stop-losses and real-time risk assessment.

5. **General Features**
   - Supports configurable flags for verbosity, enabling logs or console output.
   - Integrates a modular structure to simplify future enhancements.

## Key Components

### Sentiment Analysis and News Integration
- **Objective**: Enhance market predictions with NLP-powered sentiment scoring.
- **Models Used**: BERT, LSTM.
- **Sentiment Scoring**: Range of -1 to +1 for precise market insights.

### Technical Analysis and Strategy Development
- Combines RSI, MACD, ADX, and EMA indicators for market determination.
- Provides real-time confidence scoring and strategy refinement.

### Automated Trading Execution
- Implements configurable risk management protocols.
- Supports modular evaluation of live data for buy/sell signals.

## Directory Structure

```
MidasTechnologiesLLC/
├── assets/
│   └── MidasTechnologiesLogo.JPG
├── data/
│   └── HistoricalData.json
├── docs/
│   ├── BusinessDocumentation/
│   ├── PoliciesAndStandards/
│   ├── ManPages/
│   └── README.md
├── logs/
│   └── MidasV1.log
├── scripts/
│   └── README.md (Setup scripts and tools)
├── src/
│   ├── griffin-stuff/
│   ├── MidasV1/
│   │   ├── config/
│   │   │   └── config.config
│   │   ├── logs/
│   │   │   └── MidasV1.log
│   │   ├── modules/
│   │   │   ├── initial_checks.py
│   │   │   ├── stock_list_petitioner.py
│   │   │   └── __pycache__/
│   │   ├── tests/
│   │   │   ├── test_connection.py
│   │   │   └── test_stock_retriever.py
│   │   └── main.py
│   ├── WebScraper/
│   │   ├── data/
│   │   ├── scrapers/
│   │   └── main.py
│   └── README.md
└── README.md
```

## Standards and Best Practices

### Coding Standards
- Adheres to **PEP8** for Python.
- Modular structure ensures maintainability and scalability.

### Documentation Standards
- **Business Documentation**: Legal, corporate, and policy-related files.
- **Man Pages**: Comprehensive technical references for all modules.

### Git Standards
- **Branching Strategy**: `main` for production, `dev` for development, and feature-specific branches off `dev`.
- **Commit Messages**: Follow structured and descriptive formats.
- **Pull Requests**: Require reviewer approval for all major changes.

## Roadmap

| Phase                       | Duration   | Goals                                    |
|-----------------------------|------------|------------------------------------------|
| **Phase 1: Initial Build**  | Weeks 1-4  | Core modules, sentiment analysis, scraper |
| **Phase 2: Backtesting**    | Weeks 5-6  | Validate reliability and performance    |
| **Phase 3: Expansion**      | Weeks 7-8  | Support additional assets and strategies |
| **Phase 4: Live Trading**   | Ongoing    | Deploy trading bot and refine algorithms |

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MidasTechnologiesLLC/MidasTechnologies.git
   ```
2. **Set up the virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run tests**:
   ```bash
   pytest
   ```

## Contact

For more information, please reach out to the Midas Technologies team.

**Primary Contacts**:
- **Chief Data Officer**: Griffin 
- **Chief Technical Officer**: Collin (KleinPanic)
- **Chief Operations Officer**: Jacob 

**Note**: This project and all related files are private and for use by Midas Technologies LLC only. Unauthorized distribution or modification is strictly prohibited.

## License

For the license file, please navigate to the `docs/BusinessDocumentation/LICENSE`.

``` Author
KleinPanic
```

