# python/test_db_load.py
from data_collector import PlayerDataCollector
import pandas as pd
import numpy as np
from datetime import datetime
from db_config import engine
import os

def test_db_load():
    # Create collector instance
    collector = PlayerDataCollector()
    

    # Try loading to database
    print("\nLoading player data...")
    try:
        df = collector.load_player_stats()
    except Exception as e:
        print(f"Error loading data: {e}")
        return False


    # Print some info about the data
    print("\nData Preview:")
    print(df.describe())
    print(df.shape)
    

    # Process the data
    print("\nProcessing data...")
    try:
        players_df, performance_df = collector.process_data(df)
        
        # Print some info about the data
        print("\nPlayers DataFrame Preview:")
        print(players_df.head())
        print("\nPlayers DataFrame Info:")
        print(players_df.info())
        
        print("\nPerformance DataFrame Preview:")
        print(performance_df.head())
        print("\nPerformance DataFrame Info:")
        print(performance_df.info())
        
        # Check for missing values
        print("\nChecking for missing values in players_df:")
        print(players_df.isnull().sum())
        print("\nChecking for missing values in performance_df:")
        print(performance_df.isnull().sum())
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return False
    
    
    # Check if loading to database was successful
    print("\nLoading to database...")
    try:
        success = collector.load_to_database(players_df, performance_df)
        if success:
            print("Test successful! Data loaded to database.")
            return True
        else:
            print("Test failed! Check the error message above.")
            return False
    except Exception as e:
        print(f"Error loading to database: {e}")
        return False

if __name__ == "__main__":
    test_db_load()