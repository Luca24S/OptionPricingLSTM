import pandas as pd
from datetime import datetime
import yfinance as yf
from pandas_datareader import data as pdr

# Enable Yahoo Finance override for pandas_datareader
yf.pdr_override()

def preprocess_option_data(filepath):
    """
    Preprocesses option data from a CSV file.

    Args:
        filepath (str): Path to the CSV file containing option data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with additional columns:
                      - 'date': Parsed datetime of the option date.
                      - 'exdate': Parsed datetime of the option expiration date.
                      - 'diffDateDays': Days to expiration.
    """
    try:
        # Read CSV file
        df = pd.read_csv(filepath)

        # Convert 'date' and 'exdate' columns to datetime
        df['date'] = pd.to_datetime(df['date'], format="%m/%d/%Y", errors='coerce')
        df['exdate'] = pd.to_datetime(df['exdate'], format="%m/%d/%Y", errors='coerce')

        # Compute days to expiration
        df['diffDateDays'] = (df['exdate'] - df['date']).dt.days

        # Drop rows with invalid dates
        df.dropna(subset=['date', 'exdate', 'diffDateDays'], inplace=True)

        return df
    except Exception as e:
        raise ValueError(f"Error processing option data: {e}")


def preprocess_underlying_data(filepath):
    """
    Preprocesses underlying asset data from a CSV file.

    Args:
        filepath (str): Path to the CSV file containing underlying asset data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with additional columns:
                      - 'Simple returns': Percentage change in adjusted close price.
                      - 'Log returns': Logarithmic returns of adjusted close price.
    """
    try:
        # Read CSV file
        df = pd.read_csv(filepath)

        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Drop rows with invalid dates
        df.dropna(subset=['Date'], inplace=True)

        # Compute returns
        df['Simple returns'] = df['Adj Close'].pct_change()
        df['Log returns'] = (df['Adj Close'] / df['Adj Close'].shift(1)).apply(np.log)

        return df
    except Exception as e:
        raise ValueError(f"Error processing underlying data: {e}")


def preprocess_treasury_data(start_date=None, end_date=None):
    """
    Fetches and preprocesses U.S. Treasury data from FRED (Federal Reserve Economic Data).

    Args:
        start_date (datetime, optional): Start date for fetching data. Defaults to Jan 1, 2000.
        end_date (datetime, optional): End date for fetching data. Defaults to Dec 31, 2021.

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing treasury rates indexed by date.
    """
    if start_date is None:
        start_date = datetime(2000, 1, 1)
    if end_date is None:
        end_date = datetime(2021, 12, 31)

    try:
        # Fetch Treasury data from FRED
        df = pdr.get_data_fred('GS10', start_date, end_date)

        # Reset index for consistency
        df = df.reset_index()

        # Rename columns for clarity
        df.rename(columns={'DATE': 'Date', 'GS10': 'TreasuryValue'}, inplace=True)

        # Drop rows with missing data
        df.dropna(subset=['TreasuryValue'], inplace=True)

        return df
    except Exception as e:
        raise ValueError(f"Error fetching Treasury data: {e}")