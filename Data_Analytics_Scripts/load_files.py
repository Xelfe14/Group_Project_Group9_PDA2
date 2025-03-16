
import simfin as sf
import os

def download_shareprices():
    # Set your API-key for downloading data.
    sf.set_api_key('460e1696-925e-41ba-b556-fe23ce2cefd4')

    # Set the local directory where data-files are stored.
    data_dir = '/Users/taddeocarpinelli/Desktop/MASTER IE/Term2/Python/Python Group'
    sf.set_data_dir(data_dir)

    # Download the daily share prices data from SimFin and load into a Pandas DataFrame.
    df_shareprices = sf.load_shareprices(variant='daily')

    # Print the first few rows of share prices data.
    print("Share Prices Data (first 5 rows):")
    print(df_shareprices.head())

    return df_shareprices

def download_companies():
    # Set your API-key for downloading data.
    sf.set_api_key('460e1696-925e-41ba-b556-fe23ce2cefd4')

    # Set the local directory for companies data.
    companies_data_dir = '/Users/taddeocarpinelli/Desktop/MASTER IE/Term2/Python/Python Group/Data/Raw'
    # Ensure the directory exists.
    os.makedirs(companies_data_dir, exist_ok=True)
    sf.set_data_dir(companies_data_dir)

    # Download the companies data for the US market.
    df_companies = sf.load_companies(market='us')

    # Print the first few rows of companies data.
    print("Companies Data (first 5 rows):")
    print(df_companies.head())

    return df_companies
