"""
Data collection and processing module for player injury prevention system.
Handles loading and preprocessing of player statistics and injury data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class PlayerDataCollector:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_data_path = os.path.join(self.base_path, 'data', 'raw')
        self.processed_data_path = os.path.join(self.base_path, 'data', 'processed')
        
    def load_player_stats(self):
        """Load player statistics from Kaggle dataset"""
        file_path = os.path.join(self.raw_data_path, 'player_stats_2023.csv')
        print(f"Loading data from: {file_path}")

        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded data using {encoding} encoding")
                print(f"Loaded {len(df)} players")
                print(f"Columns: {df.columns.tolist()}")
                return df
            except UnicodeDecodeError:
                continue
          
        raise ValueError("Could not read the file with any of the attempted encodings")
        
        
if __name__ == "__main__":
    collector = PlayerDataCollector()
    df = collector.load_player_stats()
    print(df.head())