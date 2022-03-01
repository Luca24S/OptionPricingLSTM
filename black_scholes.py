import numpy as np
from scipy.stats import norm

def compute_black_scholes(option_data, underlying_data, treasury_data):
    """
    Computes Black-Scholes predicted values for option pricing.

    Args:
        option_data (pd.DataFrame): DataFrame containing option data.
        underlying_data (pd.DataFrame): DataFrame containing underlying asset data.
        treasury_data (pd.DataFrame): DataFrame containing risk-free rate data.

    Returns:
        pd.DataFrame: Updated option_data with a new column 'BS_values_predicted'.
    """

    # Merge the underlying and treasury data with option_data
    try:
        merged_data = option_data.merge(
            underlying_data[['Date', 'Adj Close']], left_on='date', right_on='Date', how='left'
        ).merge(
            treasury_data.reset_index(), left_on='date', right_on='DATE', how='left'
        )
        merged_data.rename(columns={'GS10': 'TreasuryValue'}, inplace=True)
    except Exception as e:
        raise ValueError(f"Error merging data: {e}")

    # Define the function to calculate Black-Scholes price
    def calculate_bs(row):
        try:
            S = row['Adj Close']  # Underlying price
            K = row['strike_price'] / 1000  # Strike price
            T = row['diffDateDays'] / 365  # Time to maturity
            r = row['TreasuryValue'] / 100  # Risk-free rate
            sigma = row['sigma_20_WindowDay']  # Volatility

            # Handle invalid values
            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
                return np.nan

            # Black-Scholes calculations
            d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if row['cp_flag'] == 'C':  # Call
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif row['cp_flag'] == 'P':  # Put
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:
                return np.nan
        except Exception as e:
            # Log the issue with this row if necessary
            print(f"Error computing Black-Scholes for row {row.name}: {e}")
            return np.nan

    # Apply the Black-Scholes calculation to each row
    merged_data['BS_values_predicted'] = merged_data.apply(calculate_bs, axis=1)

    # Return the updated DataFrame
    return merged_data.drop(columns=['Date', 'DATE'])