# MidasV1 Documentation 
## General Technical Overview
- Written for python.
- Needs IBJTS API and JTS running or IB gateway running. 
- Modular Nature. 

## Workflow / Program Design and Goals. 
- General: Trading bot for contracts 
1) Main.py: 
   - program ran by user that starts and controllers the other programs. 
2) Module 1 - the first barrier.
    a) Operating System Check:
        * The first part of the first module that is started by the int main. This first part / function of the first module is in charge of determing the operating system of the device running the program. By default this program will be tailored for linux systems, however, eventually, support for Windows, MacOS, BSD, and possibly illumos systems will be added. 
    b) Dependency Check: 
        * This function of module 1 will ensure that all the necessary dependencies are installed. This can be skeleton code for now as not all dependencies have been discovered or outlined. This will also be ran by int main
    c) Connectivity Check:
        * THis last function of module 1 will be called by the int mai as are all the other functions of all future modules. This module is in charge of determining if a secure connection has been established with the JTS or IB gateway program running on the user machine. It will gracefully relay these errors to the user, and determine if these errors are critical to the operation of this program, or can be ignored. 
    d) int main:
       * This function of module 1 will be the the controller of the rest of the function. This function will be called by main.py and will call all the other functions of module 1 in succession (unless it is better defined to remove the int main and just call each function indivudally through the main.py)
3) Module 2 - IBJTS List Petitioner 
   a) Scanner:
        * This overall module is harder to break down then its predicesor. This function however, should start out by taking the log content loaded by main.py The following values need to be loaded into this program, the specific volume, the net change, percent change, and more later down the line, but we'll focus on the first 3 metrics for now. These default values loaded from the config file (located in a directory called config/ inside of the project root directory - called MidasV1 in src of the entire git repo.) With these loaded values, it should use these values, and call the API to retrieve a list of STOCKS that meet the criteria loaded from the config file. This list should be cached in some way for transfer between modules, and short term storage, but it should not be stored long term if possible. 
    b) Refiner: 
         * This function / general idea of the module is to refine the data further. Here, more config information sent from the main.py which loaded the config file for this program should be sent to the refiner. Here we will refine the list to get rid of items. The first refinement shall be as follows; every item in the list, aka every stock of the list must have its individual share price recorded with it. if this stock price is larger then some value defined by the config file, then remove it from the list. the second refinement is this, but its exact logic needs to be determined through the possible API calls, but each item of the list needs to be flagged to either have possible option contracts, or not. If the item in the list does not have possible option contracts, it shall be removed from the list. The third check is comparing each item or stock's volitility index, and if it is above a value specified from the config file, it is to be removed from the list. The fourth, and currently final refinement, will check the length of the list. This refinement is conditional based on a configuration boolean that is either 1 or 0. if the boolean is false, then this refinement shall not be done, if the boolean is true, this refinement shall take place. This conditional refinement be analyzing the number of items in the overall refined list, and if it is above some value defined by the configuration file, then truncate the list to match the value specified in the config file. After this is done, this information should be packaged into the /tmp/ folder or cached, or stored in some short term manner that allows this refined list to be passed to the next module. 
4) Module 3 - Stock Information Retrieval 
   a) Load: 
      * This function or idea of this module is as follows. It should load the refined list that was created in module 2. Also it should be passed more configuration information which once again is retrieved from the configuration file. 
   b) Threaded Information Gathering and Cleaning and Choosing a Stratedgy
      * For each stock of the list, that has gone through refinement, make an indiviudal threaded procress that is responble for doing the following for each item of the list. For each stock, pull information every X minutes (retrieved from the configuration file - but it must be a recognized trading interval), the information requested is the following raw data: datetime (exclude date and time just include the single variable datetime), high, low, close, and volume. In this, if the datetime is not of a recognized trading day (Ie it is the weekend, or it is not between the times 9:30am - 4:00 pm) do not record it. Each thread should record its information to a directory called data inside the root directory, each file should be named in the convetion {stock_name}.{current_date}.json. The pulling of information should only go back 48 hours. 
      * Now based on the raw data, as it comes in, from each threated procress, should increase or decrease a counter, This counter shall be referred to as the Stratedgy Counter. Depending on the raw data, the program should algorithmically determine what indicators are the best to analyze this data for further determance of if the trade is advantagous or not. It should do this for all the information as it comes in, so the counter should change with every new input from each thread, and by the end of all the threads retreiving all the information, the counter should have an indication of what Stratedgy should be used, and that stratedgy should be chosen and passed to the next function of this module. 
    c) Stratedgy Implementation and market determination
        * after the ideal stratedgy has been decided the indicators of that stratedgy should be calculated by the code, or if more efficient, retrieved from the API. So far there is only 1 stratedgy, which shall be the default stratedgy while we improve the logic of module 3. This stratedgy involves calculating or retrieving the RSI, MACD, ADX, and EMA for each stock in the list. If the values of the RSI, MACD, ADX, or EMA, pass some threshold as defined by the config file, this will indcate market determination. So for instance, an RSI of say > 70 is an indication of a bearish market. An RSI of < 30 indicates a bullish market. MACD, if its above 0 indicates bullish, below 0 indicates a bearish market, ADX is more complicated, but generally has a range of < 20 meaning weak, between 20-25 indicates a weak movement, above 25 indicates strong movement, and if EMA is above share price, this indicates a bearish market, if its below this indcates a bearish market. Each of these indicators and all subsequent indicators will need to be weighted on their supposed affect on the market speculation of the stock / contract option. Each of these indicators, depending on their value,should add to another counter, called market counter, and depending on this value, which shall exist for each stock, each stock shall be determined to be either overall bearish, or overall bullish and a weight should be given to determine how bullish or how bearish they are. 
          * Eventualy it would be becoming to utilize system resources, and continue analysis on the complete list, however, this may be difficult in a first attempt / iteration of the program, and rate limiting may prevent this with our current plan. So there shall be an internal boolean (not needed in config) but this internal bool will be true or false. True means that the entire list will continue to the next module. False means the program should isolate the most bearish stock of the list, and the most bullish of the list. These 2 stock symbols, and any needed info shall be passed onto the next module. 
5) Module 4 - Option Chain Trading, and Risk Management
    a) Option Chain Data 
       * For the 2 stocks that are passed, labeled bullish and bearish, each of their option chain data should be pulled for the most recent experation posisble. This list should mainly be focused on the strike price for each contract, and the price of that contract. After retrieving that info for both stocks (again thread it for both of them to do it asyncrhonous and fast), the strike price should be compared to the share price. For the bearish stocks, isolate the closest strike price to share price that is cheaper than the strike price. For bearish, isolate the option contract that is selling above stock price, but is the closest selling above stock price. In the future a mechanism to change these rules based on the datetime is needed, as before 12 AM is normally prime stock market hours, so these rules may need to change depending on the time of the trade.
    b) Risk Management Stage 1: 
       * After determing the option contracts to buy, the program has to retrieve the account balance information of the user, and determine acceptable risk from the configuration file. Therewill be need to be a variable that is respobsible for this percentage, but if the contract option costs x% or less, then it is deemed as an acceptable risk, and that contract option shall continue for procressing. 
    c) Buying and Selling / Risk Managemnt 2:
        * At this stage, contract option(s) should be given to this function/section of the fourth module. This contract option(s), if it passed risk management 1, should be bought by the API (with a check to ensure the trade is only on paper). Information should continously be gathered on the bought contract option's stock's raw data as it comes in on a x timely bases (defined by the config file). Additionally, a second contract, a stop contract, should be created for a loss of x% (defined by the configuration file). 
         * From the continous information gathering, a sell stratedgy needs to be developed, much like the buy stratedgy to determine an optimal time to sell the contract after purchasing. 
6) General Additions: 
   * Add flags to the main.py to run the program with checks, without checks, verbose, etc. 
   * The program should have both print statements and log statements throughout and depending on a flag (verbose or not) the program should log this by default, or if verbose, print everything it would log to the console. Some information needs to be printed instead of logged and there will be some exceptions. 
           
           
## File Structure
* Project Directory / MidasV1: 
  * README.md 
  * requirements.txt 
  * config/ 
    * config.config
  * main.py: main script orchestrating application. 
  * modules/ 
    * data_retrieval.py 
    * analysis.py 
    * tradng_execution.py
  * tests/
    * test_data_retrieval.py 
    * test_analysis.py 
    * test_trading_execution.py 
  * logs/ 
    * MidasV1.log
