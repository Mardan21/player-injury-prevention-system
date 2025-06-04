# python/injury_integrator_synthetic.py
"""
Injury data integration pipeline with synthetic data fallback
When R scraping fails, generates realistic synthetic injury data
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import subprocess
import os
import sys
import json
from pathlib import Path
import argparse
from sqlalchemy import text

# Add python directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_config import engine

class InjuryDataIntegrator:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.r_scripts_path = self.base_path / 'r_scripts'
        self.injuries_raw_path = self.base_path / 'data' / 'injuries' / 'raw'
        self.injuries_processed_path = self.base_path / 'data' / 'injuries' / 'processed'
        
        # Create directories if they don't exist
        self.injuries_raw_path.mkdir(parents=True, exist_ok=True)
        self.injuries_processed_path.mkdir(parents=True, exist_ok=True)
        
    def setup_injury_tables(self):
        """Create injury-related database tables"""
        print("Setting up injury database tables...")
        
        injury_schema = """
        -- Drop existing injury tables
        DROP TABLE IF EXISTS injury_events CASCADE;
        DROP TABLE IF EXISTS player_injury_summary CASCADE;
        DROP TABLE IF EXISTS injury_labels CASCADE;
        
        -- Create injury_events table
        CREATE TABLE injury_events (
            injury_id SERIAL PRIMARY KEY,
            player_id INTEGER REFERENCES players(player_id),
            player_name VARCHAR(100),
            injury_date DATE,
            return_date DATE,
            days_out INTEGER,
            injury_type VARCHAR(200),
            injury_category VARCHAR(50),
            severity VARCHAR(20),
            season VARCHAR(10),
            league VARCHAR(50),
            team VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create player_injury_summary table
        CREATE TABLE player_injury_summary (
            player_id INTEGER PRIMARY KEY REFERENCES players(player_id),
            total_injuries INTEGER DEFAULT 0,
            total_days_injured INTEGER DEFAULT 0,
            injury_prone_score DECIMAL(5,3) DEFAULT 0,
            injury_frequency DECIMAL(5,2) DEFAULT 0,
            muscle_injuries INTEGER DEFAULT 0,
            knee_injuries INTEGER DEFAULT 0,
            ankle_injuries INTEGER DEFAULT 0,
            back_injuries INTEGER DEFAULT 0,
            last_injury_date DATE,
            avg_recovery_days DECIMAL(5,1),
            chronic_injury_flag BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create injury_labels table for ML
        CREATE TABLE IF NOT EXISTS injury_labels (
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
        
        -- Create indexes
        CREATE INDEX idx_injury_events_player ON injury_events(player_id);
        CREATE INDEX idx_injury_events_date ON injury_events(injury_date);
        CREATE INDEX idx_injury_events_season ON injury_events(season);
        CREATE INDEX idx_injury_summary_risk ON player_injury_summary(injury_prone_score DESC);
        CREATE INDEX idx_injury_labels_player ON injury_labels(player_id);
        """
        
        try:
            with engine.connect() as conn:
                conn.execute(text(injury_schema))
                conn.commit()
            print("✅ Injury tables created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating injury tables: {e}")
            return False
    
    def run_r_scraping(self, force_refresh=False):
        """Execute R scraping script"""
        print("Running R injury data scraping...")
        
        # Check if data already exists and force_refresh is False
        injury_file = self.injuries_processed_path / 'injuries_2023_24.csv'
        if injury_file.exists() and not force_refresh:
            print("✅ Injury data already exists. Use --force to re-scrape.")
            return True
        
        try:
            # Change to r_scripts directory
            original_dir = os.getcwd()
            os.chdir(self.r_scripts_path)
            
            # Run R script
            print("⏳ Running R scraping script (this may take 30-60 minutes)...")
            result = subprocess.run(
                ['Rscript', 'scrape_injuries.R'], 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout for testing
            )
            
            os.chdir(original_dir)
            
            if result.returncode == 0:
                print("✅ R scraping completed successfully")
                return True
            else:
                print("⚠️  R scraping failed - falling back to synthetic data")
                print("Error output:")
                print(result.stderr[:500])  # Show first 500 chars of error
                return "synthetic"  # Signal to use synthetic data
                
        except subprocess.TimeoutExpired:
            print("⚠️  R scraping timed out - falling back to synthetic data")
            return "synthetic"
        except Exception as e:
            print(f"⚠️  Error running R script - falling back to synthetic data: {e}")
            return "synthetic"
        finally:
            os.chdir(original_dir)
    
    def generate_synthetic_injury_data(self):
        """Generate realistic synthetic injury data"""
        print("🎲 Generating synthetic injury data...")
        
        try:
            # Get all players with their performance data
            with engine.connect() as conn:
                players_query = """
                SELECT 
                    p.player_id,
                    p.player_name,
                    p.age,
                    p.position,
                    p.team,
                    p.league,
                    COALESCE(pp.minutes_played, 0) as minutes_played,
                    COALESCE(pp.games_played, 0) as games_played,
                    COALESCE(pp.fouls_committed, 0) as fouls_committed,
                    COALESCE(pp.tackles, 0) as tackles,
                    COALESCE(pp.yellow_cards, 0) as yellow_cards,
                    COALESCE(pp.aerial_duels_won, 0) as aerial_duels_won
                FROM players p
                LEFT JOIN player_performance pp ON p.player_id = pp.player_id
                WHERE pp.player_id IS NOT NULL
                """
                
                players_df = pd.read_sql(players_query, conn)
            
            print(f"   Generating injuries for {len(players_df)} players...")
            
            # Define realistic injury types and categories
            injury_types = [
                ('Hamstring strain', 'muscle', 'moderate', 18),
                ('Ankle sprain', 'ankle', 'minor', 12),
                ('Knee ligament', 'knee', 'severe', 45),
                ('Calf strain', 'muscle', 'minor', 8),
                ('Groin strain', 'muscle', 'moderate', 15),
                ('Back injury', 'back', 'moderate', 20),
                ('Achilles tendon', 'ankle', 'severe', 60),
                ('Quadriceps strain', 'muscle', 'moderate', 14),
                ('Shoulder injury', 'shoulder', 'minor', 10),
                ('Concussion', 'head', 'moderate', 7),
                ('Foot fracture', 'foot', 'severe', 35),
                ('Thigh strain', 'muscle', 'minor', 6)
            ]
            
            injury_events = []
            injury_summaries = []
            injury_labels = []
            
            for _, player in players_df.iterrows():
                # Calculate injury probability based on risk factors
                base_risk = 0.12  # 12% base chance of having injury history
                
                # Risk factors
                age = player['age'] or 25
                if age > 30: base_risk += 0.08
                if age > 32: base_risk += 0.10
                
                # Position risk
                position = str(player['position']).upper()
                if any(p in position for p in ['CB', 'FB', 'LB', 'RB']): base_risk += 0.05  # Defenders
                if any(p in position for p in ['CF', 'ST', 'FW']): base_risk += 0.04       # Forwards
                
                # Workload risk
                minutes = player['minutes_played'] or 0
                if minutes > 2500: base_risk += 0.06
                if minutes > 3000: base_risk += 0.08
                
                # Physical play risk
                fouls = player['fouls_committed'] or 0
                tackles = player['tackles'] or 0
                if fouls > 40: base_risk += 0.03
                if tackles > 60: base_risk += 0.03
                
                # Determine if player has injury history
                has_injuries = np.random.random() < base_risk
                
                if has_injuries:
                    # Number of injuries (1-4, weighted toward fewer)
                    num_injuries = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05])
                    
                    total_days_out = 0
                    injury_dates = []
                    
                    for i in range(num_injuries):
                        # Select random injury type
                        injury_type, category, severity, avg_days = np.random.choice(injury_types, p=[
                            0.15, 0.12, 0.08, 0.12, 0.10, 0.08, 0.05, 0.10, 0.05, 0.03, 0.04, 0.08
                        ])
                        
                        # Generate injury date (within last 2 years)
                        days_ago = np.random.randint(30, 730)
                        injury_date = datetime.now().date() - timedelta(days=days_ago)
                        
                        # Calculate days out (with some variation)
                        days_out = max(1, int(np.random.normal(avg_days, avg_days * 0.3)))
                        return_date = injury_date + timedelta(days=days_out)
                        
                        total_days_out += days_out
                        injury_dates.append(injury_date)
                        
                        injury_events.append({
                            'player_id': player['player_id'],
                            'player_name': player['player_name'],
                            'injury_date': injury_date,
                            'return_date': return_date,
                            'days_out': days_out,
                            'injury_type': injury_type,
                            'injury_category': category,
                            'severity': severity,
                            'season': '2023-24' if injury_date >= date(2023, 7, 1) else '2022-23',
                            'league': player['league'],
                            'team': player['team']
                        })
                    
                    # Calculate injury summary
                    injury_prone_score = min(1.0, (num_injuries * 0.2 + total_days_out / 200))
                    injury_frequency = num_injuries / 2.0  # injuries per year
                    last_injury = max(injury_dates)
                    avg_recovery = total_days_out / num_injuries
                    
                    # Count by category
                    muscle_count = sum(1 for e in injury_events if e['player_id'] == player['player_id'] and e['injury_category'] == 'muscle')
                    knee_count = sum(1 for e in injury_events if e['player_id'] == player['player_id'] and e['injury_category'] == 'knee')
                    ankle_count = sum(1 for e in injury_events if e['player_id'] == player['player_id'] and e['injury_category'] == 'ankle')
                    back_count = sum(1 for e in injury_events if e['player_id'] == player['player_id'] and e['injury_category'] == 'back')
                    
                    injury_summaries.append({
                        'player_id': player['player_id'],
                        'total_injuries': num_injuries,
                        'total_days_injured': total_days_out,
                        'injury_prone_score': round(injury_prone_score, 3),
                        'injury_frequency': round(injury_frequency, 2),
                        'muscle_injuries': muscle_count,
                        'knee_injuries': knee_count,
                        'ankle_injuries': ankle_count,
                        'back_injuries': back_count,
                        'last_injury_date': last_injury,
                        'avg_recovery_days': round(avg_recovery, 1),
                        'chronic_injury_flag': num_injuries >= 3
                    })
                
                # Generate ML injury label (future risk prediction)
                future_risk = base_risk * 0.7  # Future risk slightly lower than historical
                
                # Recent injury increases future risk
                if has_injuries and injury_summaries:
                    last_injury = injury_summaries[-1]['last_injury_date']
                    days_since_injury = (datetime.now().date() - last_injury).days
                    if days_since_injury < 90:  # Recent injury
                        future_risk += 0.15
                    elif days_since_injury < 180:
                        future_risk += 0.08
                
                # Future injury prediction
                will_be_injured = np.random.random() < future_risk
                
                injury_labels.append({
                    'player_id': player['player_id'],
                    'injured_next_month': will_be_injured,
                    'injury_risk_score': round(future_risk, 3),
                    'fatigue_score': min(1.0, minutes / 3000) if minutes > 0 else 0,
                    'workload_increase': np.random.normal(0, 10),
                    'recent_injury_history': has_injuries and len(injury_summaries) > 0 and (datetime.now().date() - injury_summaries[-1]['last_injury_date']).days < 180,
                    'prediction_date': datetime.now().date()
                })
            
            print(f"   Generated {len(injury_events)} injury events")
            print(f"   Generated {len(injury_summaries)} injury summaries")  
            print(f"   Generated {len(injury_labels)} injury labels for ML")
            
            return injury_events, injury_summaries, injury_labels
            
        except Exception as e:
            print(f"❌ Error generating synthetic data: {e}")
            return None, None, None
    
    def load_synthetic_data(self, injury_events, injury_summaries, injury_labels):
        """Load synthetic data into database"""
        print("Loading synthetic injury data to database...")
        
        try:
            with engine.connect() as conn:
                # Clear existing data
                conn.execute(text("DELETE FROM injury_labels"))
                conn.execute(text("DELETE FROM player_injury_summary"))
                conn.execute(text("DELETE FROM injury_events"))
                conn.commit()
                
                # Load injury events
                if injury_events:
                    events_df = pd.DataFrame(injury_events)
                    events_df.to_sql('injury_events', engine, if_exists='append', index=False)
                    print(f"   ✅ Loaded {len(injury_events)} injury events")
                
                # Load injury summaries
                if injury_summaries:
                    summaries_df = pd.DataFrame(injury_summaries)
                    summaries_df.to_sql('player_injury_summary', engine, if_exists='append', index=False)
                    print(f"   ✅ Loaded {len(injury_summaries)} player summaries")
                
                # Load injury labels
                if injury_labels:
                    labels_df = pd.DataFrame(injury_labels)
                    labels_df.to_sql('injury_labels', engine, if_exists='append', index=False)
                    injured_count = sum(1 for label in injury_labels if label['injured_next_month'])
                    print(f"   ✅ Loaded {len(injury_labels)} ML labels ({injured_count} predicted injuries)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading synthetic data: {e}")
            return False
    
    def generate_integration_report(self, data_source="synthetic"):
        """Generate summary report of integration process"""
        print("Generating integration report...")
        
        try:
            with engine.connect() as conn:
                # Get statistics
                injury_events_count = conn.execute(text("SELECT COUNT(*) FROM injury_events")).scalar()
                injury_summaries_count = conn.execute(text("SELECT COUNT(*) FROM player_injury_summary")).scalar()
                injury_labels_count = conn.execute(text("SELECT COUNT(*) FROM injury_labels")).scalar()
                predicted_injuries = conn.execute(text("SELECT COUNT(*) FROM injury_labels WHERE injured_next_month = TRUE")).scalar()
                
                # Get top injury-prone players
                top_risk_query = """
                SELECT p.player_name, p.team, pis.injury_prone_score, pis.total_injuries
                FROM player_injury_summary pis
                JOIN players p ON pis.player_id = p.player_id
                ORDER BY pis.injury_prone_score DESC
                LIMIT 5
                """
                top_risk_players = pd.read_sql(top_risk_query, conn)
                
                # Get injury statistics by category
                injury_stats_query = """
                SELECT injury_category, COUNT(*) as count, AVG(days_out) as avg_days
                FROM injury_events
                WHERE injury_category IS NOT NULL
                GROUP BY injury_category
                ORDER BY count DESC
                """
                injury_stats = pd.read_sql(injury_stats_query, conn)
            
            # Create report
            report = {
                'integration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_source': data_source,
                'injury_events_loaded': int(injury_events_count),
                'players_with_injury_history': int(injury_summaries_count),
                'ml_labels_created': int(injury_labels_count),
                'predicted_future_injuries': int(predicted_injuries),
                'injury_rate_prediction': f"{predicted_injuries/injury_labels_count*100:.1f}%" if injury_labels_count > 0 else "0%",
                'top_injury_prone_players': top_risk_players.to_dict('records') if len(top_risk_players) > 0 else [],
                'injury_statistics_by_category': injury_stats.to_dict('records') if len(injury_stats) > 0 else []
            }
            
            # Save report
            report_file = self.injuries_processed_path / f'integration_report_{data_source}.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Integration report saved to {report_file}")
            print(f"\n📊 Integration Summary:")
            print(f"   📈 Data source: {data_source.upper()}")
            print(f"   🏥 Injury events loaded: {injury_events_count}")
            print(f"   👥 Players with injury history: {injury_summaries_count}")
            print(f"   🎯 ML labels created: {injury_labels_count}")
            print(f"   ⚠️  Predicted future injuries: {predicted_injuries} ({report['injury_rate_prediction']})")
            
            if len(top_risk_players) > 0:
                print(f"   🔴 Highest risk player: {top_risk_players.iloc[0]['player_name']} "
                      f"(Risk: {top_risk_players.iloc[0]['injury_prone_score']:.3f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return False
    
    def run_full_integration(self, force_refresh=False, use_synthetic=False):
        """Run the complete integration pipeline"""
        print("=" * 60)
        print("STARTING INJURY DATA INTEGRATION PIPELINE")
        print("=" * 60)
        
        # Step 1: Setup database
        print(f"\n📍 STEP 1: Setting up database tables")
        if not self.setup_injury_tables():
            print("❌ Integration failed at database setup")
            return False
        
        data_source = "synthetic"
        
        if not use_synthetic:
            # Step 2: Try R scraping
            print(f"\n📍 STEP 2: Running R injury data scraping")
            r_result = self.run_r_scraping(force_refresh)
            
            if r_result == True:
                data_source = "real"
                print("✅ Using real injury data from R scraping")
                # TODO: Add real data processing steps here
                print("⚠️  Real data processing not implemented yet - using synthetic data")
                data_source = "synthetic"
            elif r_result == "synthetic":
                print("⚠️  R scraping failed - proceeding with synthetic data")
                data_source = "synthetic"
        else:
            print(f"\n📍 STEP 2: Skipping R scraping (using synthetic data)")
        
        # Step 3: Generate synthetic data
        print(f"\n📍 STEP 3: Generating synthetic injury data")
        injury_events, injury_summaries, injury_labels = self.generate_synthetic_injury_data()
        
        if not injury_events:
            print("❌ Integration failed at synthetic data generation")
            return False
        
        # Step 4: Load data
        print(f"\n📍 STEP 4: Loading injury data to database")
        if not self.load_synthetic_data(injury_events, injury_summaries, injury_labels):
            print("❌ Integration failed at data loading")
            return False
        
        # Step 5: Generate report
        print(f"\n📍 STEP 5: Generating integration report")
        if not self.generate_integration_report(data_source):
            print("❌ Integration failed at report generation")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 INJURY DATA INTEGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\n🎯 Ready for ML Training!")
        print("Next steps:")
        print("1. Test ML model: python models/injury_predictor.py")
        print("2. View injury analytics: psql -d injury_prevention -c '\\dt'")
        print("3. Check injury statistics: python python/test_queries.py")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Integrate injury data (with synthetic fallback)')
    parser.add_argument('--force', action='store_true', help='Force re-scraping of injury data')
    parser.add_argument('--synthetic-only', action='store_true', help='Skip R scraping, use synthetic data only')
    parser.add_argument('--setup-only', action='store_true', help='Only setup database tables')
    
    args = parser.parse_args()
    
    integrator = InjuryDataIntegrator()
    
    if args.setup_only:
        print("Setting up injury database tables only...")
        success = integrator.setup_injury_tables()
    else:
        success = integrator.run_full_integration(
            force_refresh=args.force,
            use_synthetic=args.synthetic_only
        )
    
    if success:
        print("\n✅ Integration completed successfully!")
    else:
        print("\n❌ Integration failed. Check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)