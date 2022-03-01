from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam

def prepare_training_data(option_data, num_timesteps=60):
    """
    Prepares training data for the RNN model.
    
    Args:
        option_data (pd.DataFrame): The dataset containing features and labels.
        num_timesteps (int): Number of past time steps to include in each sample.
        
    Returns:
        X (list): List containing [time-series input, additional features].
        y (np.ndarray): Labels for the training samples.
    """
    # Extract the necessary columns
    underlying_prices = option_data['Adj Close'].values  # Time-series data
    additional_features = option_data[['strike_price', 'diffDateDays', 'TreasuryValue']].values  # Static features
    target = option_data[['best_bid', 'best_offer']].mean(axis=1).values  # Target: Average bid/ask price

    # Create sequences for the LSTM
    X_time_series = []
    X_static_features = []
    y = []

    for i in range(num_timesteps, len(underlying_prices)):
        # Create a sequence of time-series data
        X_time_series.append(underlying_prices[i - num_timesteps:i].reshape(num_timesteps, 1))
        # Include additional static features
        X_static_features.append(additional_features[i])
        # Target variable
        y.append(target[i])
    
    # Convert to numpy arrays
    X_time_series = np.array(X_time_series)
    X_static_features = np.array(X_static_features)
    y = np.array(y)

    # Combine time-series input and static features
    X = [X_time_series, X_static_features]

    return X, y

def create_lstm_model(num_timesteps, layers, features):
    """
    Creates an LSTM-based model with additional static features.
    
    Args:
        num_timesteps (int): Number of time steps in the LSTM input.
        layers (int): Number of LSTM layers.
        features (int): Number of static features.
        
    Returns:
        model: Compiled Keras model.
    """
    # Input for time-series data
    input_time_series = Input(shape=(num_timesteps, 1))

    # LSTM layers
    lstm_output = input_time_series
    for _ in range(layers):
        return_sequences = _ < layers - 1  # Return sequences for all but the last LSTM
        lstm_output = LSTM(units=8, return_sequences=return_sequences)(lstm_output)

    # Input for static features
    input_static = Input(shape=(features,))

    # Concatenate LSTM output with static features
    concat = Concatenate()([lstm_output, input_static])

    # Fully connected layers
    for _ in range(2):  # Add two dense layers
        concat = Dense(100)(concat)
        concat = LeakyReLU()(concat)

    # Output layer
    output = Dense(1, activation='relu')(concat)

    # Define the model
    model = Model(inputs=[input_time_series, input_static], outputs=output)
    return model

def train_rnn_model(option_data, num_timesteps=60, batch_size=4000, epochs=100):
    """
    Trains the RNN model using the prepared data.
    
    Args:
        option_data (pd.DataFrame): The dataset containing features and labels.
        num_timesteps (int): Number of past time steps to include in each sample.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        
    Returns:
        model: Trained Keras model.
    """
    # Prepare training data
    X, y = prepare_training_data(option_data, num_timesteps)

    # Split data into training and validation sets
    split_index = int(0.8 * len(y))  # 80% training, 20% validation
    X_train = [X[0][:split_index], X[1][:split_index]]
    y_train = y[:split_index]
    X_val = [X[0][split_index:], X[1][split_index:]]
    y_val = y[split_index:]

    # Create and compile the model
    model = create_lstm_model(num_timesteps=num_timesteps, layers=4, features=3)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

    # Train the model
    model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val)
    )

    return model