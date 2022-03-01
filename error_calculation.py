import numpy as np

def compute_errors(option_data):
    """
    Computes various error metrics for Black-Scholes predictions against actual bid-ask averages.

    Args:
        option_data (pd.DataFrame): DataFrame containing the columns:
                                    - 'best_bid': Bid prices.
                                    - 'best_offer': Offer prices.
                                    - 'BS_values_predicted': Predicted option prices.

    Returns:
        dict: A dictionary containing error metrics (MSE, RMSE, MAD, etc.).
    """
    # Calculate the bid-ask average
    option_data['bid_ask_avg'] = option_data[['best_bid', 'best_offer']].mean(axis=1)

    # Calculate the difference between actual and predicted values
    diff = option_data['bid_ask_avg'] - option_data['BS_values_predicted']

    # Handle missing or invalid data
    diff = diff.dropna()

    # Calculate error metrics
    mse = (diff ** 2).mean()  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mad = diff.abs().median()  # Median Absolute Deviation
    mae = diff.abs().mean()  # Mean Absolute Error
    mape = (diff.abs() / option_data['bid_ask_avg']).mean() * 100  # Mean Absolute Percentage Error

    # Print error metrics
    print("Error Metrics:")
    print(f"MSE  (Mean Squared Error): {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAD  (Median Absolute Deviation): {mad:.4f}")
    print(f"MAE  (Mean Absolute Error): {mae:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

    # Return error metrics as a dictionary
    errors = {
        "MSE": mse,
        "RMSE": rmse,
        "MAD": mad,
        "MAE": mae,
        "MAPE": mape,
    }

    return errors