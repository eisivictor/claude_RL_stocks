"""
DJIA Stock Trading Models Trainer

This script trains DQN trading models for all companies in the Dow Jones Industrial Average.
It imports the list_djia module to get the ticker symbols, then trains each model sequentially.
"""

import os
import time
from datetime import datetime
from stock_trading import train_model
from list_djia import get_djia_companies

def main():
    # Create a directory to store all model weights
    models_dir = "djia_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Get the list of DJIA companies
    print("Fetching list of Dow Jones Industrial Average companies...")
    djia_tickers = get_djia_companies()
    
    # Display the tickers we'll be training on
    print(f"Found {len(djia_tickers)} companies in the DJIA:")
    print(", ".join(djia_tickers))
    
    # Define training parameters
    training_params = {
        "lookback": 20,
        "gamma": 0.99,
        "batch_size": 64,
        "learning_rate": 0.001,
        "epsilon_initial": 1.0,
        "epsilon_final": 0.01,
        "epsilon_decay": 10000,
        "memory_size": 10000,
        "episodes": 40,
        "initial_capital": 10000,
        "period": 1
    }
    
    # Log file setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(models_dir, f"training_log_{timestamp}.txt")
    
    # Train models for each ticker
    successful_tickers = []
    failed_tickers = []
    
    with open(log_file, "w") as log:
        log.write(f"DJIA Model Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Total companies to process: {len(djia_tickers)}\n\n")
        
        for i, ticker in enumerate(djia_tickers):
            start_time = time.time()
            print(f"\n[{i+1}/{len(djia_tickers)}] Training model for {ticker}...")
            log.write(f"\n--- {ticker} ---\n")
            log.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            try:
                # Train the model for this ticker
                agent, data = train_model(
                    ticker=ticker,
                    **training_params
                )
                
                # Move the model file to our organized directory
                source_file = f"{ticker.lower()}_trading_model.weights.h5"
                target_file = os.path.join(models_dir, source_file)
                
                if os.path.exists(source_file):
                    # If file exists in current directory, move it to models directory
                    if os.path.exists(target_file):
                        os.remove(target_file)  # Remove existing file in target location
                    os.rename(source_file, target_file)
                
                # Record success
                elapsed_time = time.time() - start_time
                successful_tickers.append(ticker)
                log.write(f"Status: Success\n")
                log.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
                print(f"✓ Successfully trained model for {ticker} in {elapsed_time:.2f} seconds")
                
            except Exception as e:
                # Record failure
                elapsed_time = time.time() - start_time
                failed_tickers.append((ticker, str(e)))
                log.write(f"Status: Failed\n")
                log.write(f"Error: {str(e)}\n")
                log.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
                print(f"✗ Failed to train model for {ticker}: {str(e)}")
        
        # Write summary
        log.write(f"\n\n--- Training Summary ---\n")
        log.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Successful: {len(successful_tickers)} companies\n")
        log.write(f"Failed: {len(failed_tickers)} companies\n")
        
        if failed_tickers:
            log.write("\nFailed companies:\n")
            for ticker, error in failed_tickers:
                log.write(f"- {ticker}: {error}\n")
    
    # Print summary
    print("\n--- Training Summary ---")
    print(f"Total companies processed: {len(djia_tickers)}")
    print(f"Successfully trained: {len(successful_tickers)}")
    print(f"Failed: {len(failed_tickers)}")
    
    if successful_tickers:
        print(f"\nSuccessful tickers: {', '.join(successful_tickers)}")
    
    if failed_tickers:
        print("\nFailed tickers:")
        for ticker, error in failed_tickers:
            print(f"- {ticker}: {error}")
    
    print(f"\nAll model weights saved to '{models_dir}/' directory")
    print(f"Training log saved to '{log_file}'")

if __name__ == "__main__":
    main()