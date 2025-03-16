

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def process_data():


    # Sub-Part 1: Loading & Inspecting the Company Dataset

    print("Loading Company Dataset...")
    df_comp = pd.read_csv('../Data/Raw/us-companies.csv', sep=';')

    # Drop rows with no Ticker
    df_comp = df_comp.dropna(subset=['Ticker'])
    # Drop columns not needed
    df_comp = df_comp.drop(columns="Business Summary")
    df_comp = df_comp.drop(
        columns=[
            "ISIN", "Number Employees", "CIK", "Market",
            "IndustryId", "End of financial year (month)",
            "Main Currency"
        ]
    )
    print("\nCompany Data Columns After Cleanup:", df_comp.columns.to_list())


    # Sub-Part 2: Loading & Inspecting the Share Price Dataset

    print("\nLoading Share Price Dataset...")
    df_share = pd.read_csv('../Data/Raw/us-shareprices-daily.csv', sep=';')



    # Drop columns with excessive missing values (e.g., Dividend)
    if 'Dividend' in df_share.columns:
        df_share = df_share.drop(columns="Dividend")

    # Forward-fill missing 'Shares Outstanding'
    df_share['Shares Outstanding'] = df_share['Shares Outstanding'].ffill()


    # Sub-Part 3: Merging Both Datasets

    print("\nMerging Datasets...")
    df_merged = pd.merge(
        df_share,
        df_comp,
        on='SimFinId',
        how='left'
    )




    # Drop redundant Ticker column and rename
    if 'Ticker_y' in df_merged.columns:
        df_merged.drop(columns=['Ticker_y'], inplace=True)
    if 'Ticker_x' in df_merged.columns:
        df_merged.rename(columns={'Ticker_x': 'Ticker'}, inplace=True)
    selected_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
    df_merged = df_merged[df_merged['Ticker'].isin(selected_tickers)]
    # Convert 'Date' to datetime
    df_merged['Date'] = pd.to_datetime(df_merged['Date'])


    # Sub-Part 4: Deriving Time-Based & Technical Features

    print("\nCreating Time-Based & Technical Features...")
    # 1) Time-based features
    df_merged['day_of_week'] = df_merged['Date'].dt.dayofweek
    df_merged['month'] = df_merged['Date'].dt.month
    df_merged['quarter'] = df_merged['Date'].dt.quarter
    df_merged['year'] = df_merged['Date'].dt.year

    # 2) Daily return
    df_merged['prev_close'] = df_merged.groupby('Ticker')['Close'].shift(1)
    df_merged['daily_return'] = ((df_merged['Close'] - df_merged['prev_close']) / df_merged['prev_close']) * 100
    df_merged.drop(columns=['prev_close'], inplace=True)

    # 3) Exponential moving average
    df_merged['ema_20'] = df_merged.groupby('Ticker')['Close'].transform(
        lambda x: x.ewm(span=20, adjust=False).mean()
    )

    # 4) Next-day return, classification target, next-day price
    df_merged['next_day_return'] = df_merged.groupby('Ticker')['daily_return'].shift(-1)
    df_merged['target_class'] = (df_merged['next_day_return'] > 0).astype(int)
    df_merged['next_day_price'] = df_merged['Close'].shift(-1)

    # Reset index
    df_merged.reset_index(drop=True, inplace=True)


    # Sub-Part 5: Handling Infinite Values & Final Cleanup

    print("\nHandling Infinite Values & Final Cleanup...")

    # Check for infinite values
    import numpy as np
    for col in ["daily_return", "next_day_return"]:
        inf_count = np.isinf(df_merged[col]).sum()
        print(f"Column '{col}' has {inf_count} infinite values initially.")

    # Replace inf with NaN and forward-fill
    df_merged['daily_return'] = df_merged['daily_return'].replace([np.inf, -np.inf], np.nan)
    df_merged['next_day_return'] = df_merged['next_day_return'].replace([np.inf, -np.inf], np.nan)

    df_merged['daily_return'] = df_merged.groupby('Ticker')['daily_return'].ffill()
    df_merged['next_day_return'] = df_merged.groupby('Ticker')['next_day_return'].ffill()

    # Drop rows still missing daily_return or next_day_return
    df_merged.dropna(subset=['daily_return', 'next_day_return', 'next_day_price'], inplace=True)


    # Final drop of columns before saving
    drop_cols = ['Open','High','Low','Adj. Close']
    for c in drop_cols:
        if c in df_merged.columns:
            df_merged.drop(columns=c, inplace=True)

    # Save final dataset
    output_dir = '../Streamlit_app/data'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'merged_features.csv')
    df_merged.to_csv(output_file, index=False)

    print("\nFinal dataset saved to:", output_file)
    print("Data processing complete!")

    return df_merged

if __name__ == '__main__':
    process_data()
