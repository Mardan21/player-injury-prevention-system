# test_db_connection.py
# Simple database connection test

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

try:
    from db_config import engine
    print("✅ Database configuration loaded successfully")
except ImportError as e:
    print(f"❌ Database config error: {e}")
    exit(1)

import pandas as pd
from sqlalchemy import text

def test_basic_connection():
    """Test basic database connection"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            print("✅ Database connection successful")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_tables_exist():
    """Check if required tables exist"""
    tables_to_check = ['players', 'player_performance', 'injury_events']
    
    try:
        with engine.connect() as conn:
            for table in tables_to_check:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"✅ Table '{table}': {count} records")
        return True
    except Exception as e:
        print(f"❌ Table check failed: {e}")
        return False

def test_simple_query():
    """Test a simple query"""
    try:
        query = "SELECT player_id, player_name FROM players LIMIT 5"
        df = pd.read_sql(query, engine)
        print(f"✅ Simple query successful: {len(df)} rows")
        print(df.head())
        return True
    except Exception as e:
        print(f"❌ Simple query failed: {e}")
        return False

def test_injury_data():
    """Check injury data"""
    try:
        query = "SELECT COUNT(*) as total, MIN(injury_date) as min_date, MAX(injury_date) as max_date FROM injury_events"
        df = pd.read_sql(query, engine)
        print(f"✅ Injury data check:")
        print(f"   Total injuries: {df.iloc[0]['total']}")
        print(f"   Date range: {df.iloc[0]['min_date']} to {df.iloc[0]['max_date']}")
        return True
    except Exception as e:
        print(f"❌ Injury data check failed: {e}")
        return False

if __name__ == "__main__":
    print("=== DATABASE CONNECTION DIAGNOSTIC ===")
    
    if not test_basic_connection():
        exit(1)
    
    if not test_tables_exist():
        exit(1)
        
    if not test_simple_query():
        exit(1)
        
    if not test_injury_data():
        exit(1)
    
    print("\n✅ All database tests passed!")
    print("Database is ready for ML model training.")