#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Neural Network and Deep Learning
# Luca Sanfilippo
# Title: Option Pricing with a Neural Network Approach
# 31 May 2022

import logging
from data_preprocessing import preprocess_option_data, preprocess_underlying_data, preprocess_treasury_data
from black_scholes import compute_black_scholes
from error_calculation import compute_errors
from rnn_model import train_rnn_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
OPTION_DATA_PATH = "data/option_data.csv"
UNDERLYING_DATA_PATH = "data/underlying_data.csv"
TREASURY_START_DATE = "2000-01-01"
TREASURY_END_DATE = "2021-12-31"

def main():
    """
    Main function to run the option pricing project pipeline.
    """
    logging.info("Starting Option Pricing Project with Neural Networks.")

    try:
        # Step 1: Data Preprocessing
        logging.info("Preprocessing option data from file: %s", OPTION_DATA_PATH)
        option_data = preprocess_option_data(OPTION_DATA_PATH)

        logging.info("Preprocessing underlying data from file: %s", UNDERLYING_DATA_PATH)
        underlying_data = preprocess_underlying_data(UNDERLYING_DATA_PATH)

        logging.info("Fetching treasury data from %s to %s", TREASURY_START_DATE, TREASURY_END_DATE)
        treasury_data = preprocess_treasury_data(
            start_date=TREASURY_START_DATE,
            end_date=TREASURY_END_DATE
        )

        # Step 2: Compute Black-Scholes Values
        logging.info("Computing Black-Scholes predicted values...")
        option_data = compute_black_scholes(option_data, underlying_data, treasury_data)

        # Step 3: Calculate Errors
        logging.info("Calculating prediction errors...")
        error_metrics = compute_errors(option_data)

        logging.info("Error Metrics:")
        for metric, value in error_metrics.items():
            logging.info(f"{metric}: {value:.4f}")

        # Step 4: Train RNN Model
        logging.info("Training the RNN model...")
        trained_model = train_rnn_model(option_data)

        logging.info("RNN model training completed. Model is ready.")

    except Exception as e:
        logging.error("An error occurred during the pipeline execution.")
        logging.exception(e)
        raise

    logging.info("Project completed successfully.")

if __name__ == "__main__":
    main()