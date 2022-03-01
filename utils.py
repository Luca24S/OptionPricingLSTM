import numpy as np
import tensorflow as tf

def scale_data(df, columns):
    """
    Scales the specified columns of a DataFrame to the range [0, 1] using TensorFlow.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to scale.
        columns (list): List of column names to scale.
        
    Returns:
        scaled_df (pd.DataFrame): DataFrame with scaled columns.
        scalers (dict): Dictionary containing the scaling parameters for each column.
    """
    scalers = {}
    scaled_values = {}

    for col in columns:
        # Convert column to TensorFlow tensor
        values = tf.convert_to_tensor(df[col].values, dtype=tf.float32)
        
        # Calculate min and max for scaling
        min_val = tf.reduce_min(values)
        max_val = tf.reduce_max(values)
        
        # Apply min-max scaling
        scaled = (values - min_val) / (max_val - min_val)
        
        # Save scaling parameters for later use
        scalers[col] = {'min': min_val.numpy(), 'max': max_val.numpy()}
        scaled_values[col] = scaled.numpy()

    # Replace original columns with scaled values
    scaled_df = df.copy()
    for col in columns:
        scaled_df[col] = scaled_values[col]
    
    return scaled_df, scalers

def inverse_scale(scaled_values, scaler):
    """
    Reverts scaled values back to their original range using scaling parameters.
    
    Args:
        scaled_values (np.ndarray): Array of scaled values.
        scaler (dict): Dictionary containing `min` and `max` values used during scaling.
        
    Returns:
        original_values (np.ndarray): Reverted values in the original scale.
    """
    return scaled_values * (scaler['max'] - scaler['min']) + scaler['min']