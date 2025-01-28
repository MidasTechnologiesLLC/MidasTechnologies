import os
import pandas as pd

def clean_tickers(data_dir):
    # Define paths
    input_file = os.path.join(data_dir, "us_tickers.csv")
    output_dir = os.path.join(data_dir, "csv")
    output_cleaned = os.path.join(output_dir, "cleaned_us_tickers.csv")
    output_refined = os.path.join(output_dir, "refined_us_tickers.csv")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found in {data_dir}. Please ensure the file is in the correct directory.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the CSV file
    try:
        tickers_data = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Step 1: Initial Cleanup
    relevant_columns = ['Ticker', 'Name', 'Primary Exchange', 'Type']
    cleaned_data = tickers_data[relevant_columns]

    # Filter rows with specific types (e.g., CS for common stock, ETF for exchange-traded fund)
    valid_types = ['CS', 'ETF']
    cleaned_data = cleaned_data[cleaned_data['Type'].isin(valid_types)]

    # Drop rows with missing values in critical columns
    cleaned_data = cleaned_data.dropna(subset=['Ticker', 'Name', 'Primary Exchange'])

    # Save the cleaned data to a new file
    cleaned_data.to_csv(output_cleaned, index=False)
    print(f"Cleaned data saved to {output_cleaned}.")

    # Step 2: Ask for further refinement
    refine = input("Do you want to refine further to keep only Ticker IDs (formatted as CSV)? (yes/no): ").strip().lower()
    if refine == "yes":
        # Keep only the Ticker column and format as a single line CSV
        refined_data = cleaned_data[['Ticker']]
        ticker_list = refined_data['Ticker'].tolist()
        
        # Save as a single line CSV
        with open(output_refined, 'w') as f:
            f.write(','.join(ticker_list))
        
        print(f"Refined data saved to {output_refined} (formatted as a single-line CSV).")
    else:
        print("No further refinement done. Exiting.")

def main():
    # Define the data directory
    data_dir = "data"  # Relative path to the data directory

    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Run the cleaning process
    clean_tickers(data_dir)

if __name__ == "__main__":
    main()

