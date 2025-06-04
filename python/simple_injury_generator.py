"""
Simple, bulletproof injury data generator
Compatible with different database configuration styles
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import sys
import os

# Add python directory to path
sys.path.append('python')

# Import your existing database configuration
try:
    from db_config import engine
    from sqlalchemy import text
except ImportError as e:
    print(f"❌ Could not import database config: {e}")
    exit(1)

def get_connection():
    """Get database connection using SQLAlchemy engine"""
    return engine.connect()

def create_injury_tables():
    """Create injury tables in database"""
    print("🏗️  Setting up injury tables...")
    
    try:
        with get_connection() as conn:
            # Create injury_labels table for ML
            conn.execute(text("""
                DROP TABLE IF EXISTS injury_labels CASCADE;
                
                CREATE TABLE injury_labels (
                    label_id SERIAL PRIMARY KEY,
                    player_id INTEGER REFERENCES players(player_id),
                    injured_next_month BOOLEAN DEFAULT FALSE,
                    injury_risk_score DECIMAL(4,3) DEFAULT 0,
                    fatigue_score DECIMAL(4,3) DEFAULT 0,
                    workload_increase DECIMAL(6,2) DEFAULT 0,
                    recent_injury_history BOOLEAN DEFAULT FALSE,
                    prediction_date DATE DEFAULT CURRENT_DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_injury_labels_player ON injury_labels(player_id);
            """))
            conn.commit()
        
        print("✅ Injury tables created")
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def generate_synthetic_injuries():
    """Generate realistic synthetic injury predictions"""
    print("🎲 Generating synthetic injury data...")
    
    try:
        # Get all players with performance data using SQLAlchemy
        query = text("""
        SELECT 
            p.player_id,
            p.player_name,
            COALESCE(p.age, 25) as age,
            COALESCE(p.position, 'MF') as position,
            COALESCE(pp.minutes_played, 0) as minutes_played,
            COALESCE(pp.games_played, 0) as games_played,
            COALESCE(pp.fouls_committed, 0) as fouls_committed,
            COALESCE(pp.tackles, 0) as tackles,
            COALESCE(pp.yellow_cards, 0) as yellow_cards
        FROM players p
        LEFT JOIN player_performance pp ON p.player_id = pp.player_id
        WHERE pp.player_id IS NOT NULL
        """)
        
        with get_connection() as conn:
            df = pd.read_sql(query, conn)
        
        print(f"   Found {len(df)} player records")
        
        if len(df) == 0:
            print("❌ No player performance data found!")
            return False
        
        injury_labels = []
        
        print("   Calculating injury risks...")
        for i, player in df.iterrows():
            if i % 500 == 0 and i > 0:
                print(f"   Processed {i}/{len(df)} players...")
            
            # Calculate realistic injury risk
            risk = 0.12  # Base 12% injury risk
            
            # Age factor
            try:
                age = int(float(player['age']))
                if age > 30:
                    risk += 0.08
                if age > 32:
                    risk += 0.10
            except (ValueError, TypeError):
                age = 25  # Default age
            
            # Position factor (defenders and forwards have more contact)
            try:
                position = str(player['position']).upper()
                if any(pos in position for pos in ['CB', 'LB', 'RB', 'FB']):
                    risk += 0.05  # Defenders
                if any(pos in position for pos in ['CF', 'ST', 'FW']):
                    risk += 0.04  # Forwards
            except (ValueError, TypeError):
                position = 'MF'
            
            # Workload factor
            try:
                minutes = int(float(player['minutes_played']))
                if minutes > 2500:
                    risk += 0.08
                if minutes > 3000:
                    risk += 0.12
            except (ValueError, TypeError):
                minutes = 0
            
            # Physical play factor
            try:
                fouls = int(float(player['fouls_committed']))
                tackles = int(float(player['tackles']))
                if fouls > 50:
                    risk += 0.04
                if tackles > 80:
                    risk += 0.03
            except (ValueError, TypeError):
                fouls = tackles = 0
            
            # Add randomness
            risk += np.random.normal(0, 0.06)
            
            # Keep risk between 0.02 and 0.80
            risk = max(0.02, min(0.80, risk))
            
            # Determine if injured
            will_be_injured = np.random.random() < risk
            
            # Calculate other factors
            fatigue = min(1.0, minutes / 3000) if minutes > 0 else 0
            workload_change = np.random.normal(0, 12)  # % change
            recent_injury = np.random.choice([True, False], p=[0.2, 0.8])
            
            injury_labels.append({
                'player_id': int(player['player_id']),
                'injured_next_month': bool(will_be_injured),
                'injury_risk_score': round(float(risk), 3),
                'fatigue_score': round(float(fatigue), 3),
                'workload_increase': round(float(workload_change), 2),
                'recent_injury_history': bool(recent_injury),
                'prediction_date': date.today()
            })
        
        print(f"   Generated {len(injury_labels)} injury risk assessments")
        
        # Convert to DataFrame and use pandas to_sql for bulk insert
        labels_df = pd.DataFrame(injury_labels)
        
        # Load into database using pandas (easier with SQLAlchemy)
        with get_connection() as conn:
            # Clear existing data
            conn.execute(text("DELETE FROM injury_labels"))
            conn.commit()
            print("   Cleared existing injury labels")
            
            # Insert new data using pandas
            labels_df.to_sql('injury_labels', conn, if_exists='append', index=False)
            print("   ✅ Loaded injury labels to database")
        
        # Show statistics
        injured_count = sum(1 for label in injury_labels if label['injured_next_month'])
        injury_rate = injured_count / len(injury_labels) * 100
        avg_risk = sum(l['injury_risk_score'] for l in injury_labels) / len(injury_labels)
        
        print(f"\n📊 INJURY DATA SUMMARY:")
        print(f"   Total players: {len(injury_labels)}")
        print(f"   Predicted injuries: {injured_count}")
        print(f"   Injury rate: {injury_rate:.1f}%")
        print(f"   Average risk score: {avg_risk:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating synthetic data: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_data():
    """Verify the generated data"""
    print("\n🔍 Verifying generated data...")
    
    try:
        with get_connection() as conn:
            # Check injury_labels table
            result = conn.execute(text("SELECT COUNT(*) FROM injury_labels"))
            total_labels = result.scalar()
            
            result = conn.execute(text("SELECT COUNT(*) FROM injury_labels WHERE injured_next_month = TRUE"))
            injured_count = result.scalar()
            
            result = conn.execute(text("SELECT AVG(injury_risk_score) FROM injury_labels"))
            avg_risk = result.scalar()
            
            print(f"✅ Verification Results:")
            print(f"   Labels in database: {total_labels}")
            print(f"   Predicted injuries: {injured_count}")
            print(f"   Average risk: {float(avg_risk):.3f}")
            
            if total_labels > 0 and injured_count > 0:
                print(f"✅ Data looks good for ML training!")
                return True
            else:
                print(f"⚠️  Data may have issues")
                return False
                
    except Exception as e:
        print(f"❌ Error verifying data: {e}")
        return False

def main():
    """Run the complete synthetic injury generation"""
    print("🏥 SYNTHETIC INJURY DATA GENERATOR")
    print("=" * 50)
    print("✅ Using your existing SQLAlchemy database configuration")
    
    try:
        # Step 1: Create tables
        if not create_injury_tables():
            return False
        
        # Step 2: Generate synthetic data
        if not generate_synthetic_injuries():
            return False
            
        # Step 3: Verify data
        if not verify_data():
            return False
        
        print(f"\n🎉 SUCCESS! Injury prediction data ready for ML training")
        print(f"Next step: python models/injury_predictor.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)