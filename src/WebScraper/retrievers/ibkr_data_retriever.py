import json
from datetime import datetime
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBKRDataRetriever(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def connect_and_retrieve_data(self):
        self.connect("127.0.0.1", 7497, clientId=0)  # Ensure IB Gateway or TWS is running
        contract = Contract()
        contract.symbol = "AAPL"  # Example stock; replace as needed
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        self.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime=datetime.now().strftime("%Y%m%d %H:%M:%S"),
            durationStr="1 D",
            barSizeSetting="1 day",
            whatToShow="MIDPOINT",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

    def historicalData(self, reqId, bar):
        data = {
            "Date": bar.date,
            "Close/Last": bar.close,
            "Volume": bar.volume,
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low
        }
        self.save_data_to_json(data)

    def save_data_to_json(self, data):
        json_path = "../data/HistoricalData.json"
        try:
            with open(json_path, "r") as file:
                historical_data = json.load(file)
        except FileNotFoundError:
            historical_data = []

        historical_data.insert(0, data)
        
        with open(json_path, "w") as file:
            json.dump(historical_data, file, indent=4)
        print(f"Data saved to {json_path}")

if __name__ == "__main__":
    app = IBKRDataRetriever()
    app.connect_and_retrieve_data()

