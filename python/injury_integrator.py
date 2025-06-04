# python/injury_integrator.py
"""
Injury data integration pipeline - Bridge between R scraping and Python/Database
Processes R outputs and loads injury data into PostgreSQL
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
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
        
        -- Create indexes
        CREATE INDEX idx_injury_events_player ON injury_events(player_id);
        CREATE INDEX idx_injury_events_date ON injury_events(injury_date);
        CREATE INDEX idx_injury_events_season ON injury_events(season);
        CREATE INDEX idx_injury_summary_risk ON player_injury_summary(injury_prone_score DESC);
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
        injury_file = self.injuries_processed_path / 'injuries_2022_23.csv'
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
                timeout=7200  # 2 hour timeout
            )
            
            os.chdir(original_dir)
            
            if result.returncode == 0:
                print("✅ R scraping completed successfully")
                print("Script output:")
                print(result.stdout)
                return True
            else:
                print("❌ R scraping failed")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ R scraping timed out (took longer than 2 hours)")
            return False
        except Exception as e:
            print(f"❌ Error running R script: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def process_injury_data(self):
        """Process raw injury data from R outputs"""
        print("Processing injury data...")
        
        try:
            # Load processed injury data from R
            injury_file = self.injuries_processed_path / 'injuries_2022_23.csv'
            
            if not injury_file.exists():
                print(f"❌ Injury data file not found: {injury_file}")
                return None
            
            df_injuries = pd.read_csv(injury_file)
            print(f"✅ Loaded {len(df_injuries)} injury records")
            
            # Clean and standardize data
            df_injuries_clean = self._clean_injury_data(df_injuries)
            
            # Save cleaned data
            cleaned_file = self.injuries_processed_path / 'injuries_cleaned.csv'
            df_injuries_clean.to_csv(cleaned_file, index=False)
            print(f"✅ Cleaned injury data saved to {cleaned_file}")
            
            return df_injuries_clean
            
        except Exception as e:
            print(f"❌ Error processing injury data: {e}")
            return None
    
    def _clean_injury_data(self, df):
        """Clean and standardize injury data"""
        print("   Cleaning injury data...")
        
        # Standardize column names
        df_clean = df.copy()
        
        # CRITICAL FIX: Clean the days_out column (remove "days" text and convert to int)
        if 'days_out' in df_clean.columns:
            df_clean['days_out'] = (
                df_clean['days_out']
                .astype(str)
                .str.replace(' days', '', regex=False)
                .str.replace(' day', '', regex=False)
                .str.strip()
            )
            # Convert to numeric, handle any non-numeric values
            df_clean['days_out'] = pd.to_numeric(df_clean['days_out'], errors='coerce')
        
        # Convert dates
        date_columns = ['injury_date_clean', 'return_date_clean', 'injury_from', 'injury_until', 'injury_date', 'return_date']
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Standardize player names for matching
        if 'player_name' in df_clean.columns:
            df_clean['player_name_normalized'] = (
                df_clean['player_name']
                .str.strip()
                .str.title()
                .str.replace(r'[^\w\s]', '', regex=True)
            )
        
        # Ensure required columns exist with proper defaults
        required_cols = ['injury_date', 'return_date', 'days_out', 'injury_type', 
                        'injury_category', 'severity', 'season']
        
        for col in required_cols:
            if col not in df_clean.columns:
                df_clean[col] = None
        
        # Filter for valid data (NOW days_out is numeric)
        df_clean = df_clean[df_clean['days_out'].notna() & (df_clean['days_out'] > 0)]
        
        print(f"   ✅ Cleaned data: {len(df_clean)} valid injury records")
        return df_clean
    
    def match_players_to_database(self, df_injuries):
        """Match injury data players to existing database players"""
        print("Matching players to database...")
        
        try:
            # Get existing players from database
            with engine.connect() as conn:
                db_players = pd.read_sql(
                    "SELECT player_id, player_name FROM players", 
                    conn
                )
            
            # Normalize player names for matching
            db_players['player_name_normalized'] = (
                db_players['player_name']
                .str.strip()
                .str.title()
                .str.replace(r'[^\w\s]', '', regex=True)
            )
            
            # Match by normalized names
            df_matched = df_injuries.merge(
                db_players[['player_id', 'player_name_normalized']],
                on='player_name_normalized',
                how='left'
            )
            
            matched_count = df_matched['player_id'].notna().sum()
            total_count = len(df_matched)
            
            print(f"✅ Matched {matched_count}/{total_count} injury records to database players")
            print(f"   Match rate: {matched_count/total_count:.1%}")
            
            # Only keep matched records
            df_matched = df_matched[df_matched['player_id'].notna()]
            
            return df_matched
            
        except Exception as e:
            print(f"❌ Error matching players: {e}")
            return None
    
    def load_injury_events(self, df_injuries):
        """Load injury events into database"""
        print("Loading injury events to database...")
        
        try:
            # Prepare data for database
            injury_events = df_injuries[[
                'player_id', 'player_name_normalized', 'injury_date', 'return_date',
                'days_out', 'injury_type', 'injury_category', 'severity', 'season'
            ]].copy()
            
            # Rename columns to match database
            injury_events = injury_events.rename(columns={
                'player_name_normalized': 'player_name'
            })
            
            # Add league and team info (get from players table)
            with engine.connect() as conn:
                player_info = pd.read_sql("""
                    SELECT player_id, team, league 
                    FROM players 
                    WHERE player_id IN ({})
                """.format(','.join(map(str, injury_events['player_id'].unique()))), conn)
            
            injury_events = injury_events.merge(player_info, on='player_id', how='left')
            
            # Clear existing injury events
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM injury_events"))
                conn.commit()
            
            # Load to database
            injury_events.to_sql('injury_events', engine, if_exists='append', index=False)
            
            print(f"✅ Loaded {len(injury_events)} injury events to database")
            return True
            
        except Exception as e:
            print(f"❌ Error loading injury events: {e}")
            return False
    
    def calculate_injury_summaries(self):
        """Calculate player injury summary statistics"""
        print("Calculating player injury summaries...")
        
        summary_query = """
        WITH injury_stats AS (
            SELECT 
                player_id,
                COUNT(*) as total_injuries,
                SUM(days_out) as total_days_injured,
                AVG(days_out) as avg_recovery_days,
                MAX(injury_date) as last_injury_date,
                
                -- Count by injury type
                COUNT(*) FILTER (WHERE injury_category IN ('muscle', 'hamstring', 'calf', 'groin')) as muscle_injuries,
                COUNT(*) FILTER (WHERE injury_category = 'knee') as knee_injuries,
                COUNT(*) FILTER (WHERE injury_category = 'ankle') as ankle_injuries,
                COUNT(*) FILTER (WHERE injury_category = 'back') as back_injuries,
                
                -- FIXED: Calculate injury frequency (injuries per year) - simplified approach
                CASE 
                    WHEN MAX(injury_date) IS NOT NULL AND MIN(injury_date) IS NOT NULL 
                        AND MAX(injury_date) != MIN(injury_date)
                    THEN COUNT(*) * 365.0 / NULLIF(MAX(injury_date) - MIN(injury_date), 0)
                    ELSE COUNT(*) * 1.0  -- If only one injury or same date, assume 1 per year
                END as injury_frequency,
                
                -- Chronic injury flag (3+ injuries)
                CASE WHEN COUNT(*) >= 3 THEN TRUE ELSE FALSE END as chronic_injury_flag
                
            FROM injury_events
            GROUP BY player_id
        ),
        injury_risk_scores AS (
            SELECT 
                *,
                -- Calculate injury prone score (0-1 scale)
                LEAST(1.0, 
                    (total_injuries * 0.3 + 
                    total_days_injured / 365.0 * 0.4 + 
                    COALESCE(injury_frequency, 0) * 0.3)
                ) as injury_prone_score
                
            FROM injury_stats
        )
        INSERT INTO player_injury_summary (
            player_id, total_injuries, total_days_injured, injury_prone_score,
            injury_frequency, muscle_injuries, knee_injuries, ankle_injuries,
            back_injuries, last_injury_date, avg_recovery_days, chronic_injury_flag
        )
        SELECT 
            player_id, total_injuries, total_days_injured, injury_prone_score,
            injury_frequency, muscle_injuries, knee_injuries, ankle_injuries,
            back_injuries, last_injury_date, avg_recovery_days, chronic_injury_flag
        FROM injury_risk_scores
        ON CONFLICT (player_id) DO UPDATE SET
            total_injuries = EXCLUDED.total_injuries,
            total_days_injured = EXCLUDED.total_days_injured,
            injury_prone_score = EXCLUDED.injury_prone_score,
            injury_frequency = EXCLUDED.injury_frequency,
            muscle_injuries = EXCLUDED.muscle_injuries,
            knee_injuries = EXCLUDED.knee_injuries,
            ankle_injuries = EXCLUDED.ankle_injuries,
            back_injuries = EXCLUDED.back_injuries,
            last_injury_date = EXCLUDED.last_injury_date,
            avg_recovery_days = EXCLUDED.avg_recovery_days,
            chronic_injury_flag = EXCLUDED.chronic_injury_flag,
            updated_at = CURRENT_TIMESTAMP
        """
        
        try:
            with engine.connect() as conn:
                conn.execute(text(summary_query))
                conn.commit()
                
                # Get count of summaries created
                result = conn.execute(text("SELECT COUNT(*) FROM player_injury_summary"))
                count = result.scalar()
                
            print(f"✅ Created injury summaries for {count} players")
            return True
            
        except Exception as e:
            print(f"❌ Error calculating injury summaries: {e}")
            return False
    
    def generate_integration_report(self):
        """Generate summary report of integration process"""
        print("Generating integration report...")
        
        try:
            with engine.connect() as conn:
                # Get statistics
                injury_events_count = conn.execute(text("SELECT COUNT(*) FROM injury_events")).scalar()
                injury_summaries_count = conn.execute(text("SELECT COUNT(*) FROM player_injury_summary")).scalar()
                
                # Get top injury-prone players
                top_risk_query = """
                SELECT p.player_name, p.team, pis.injury_prone_score, pis.total_injuries
                FROM player_injury_summary pis
                JOIN players p ON pis.player_id = p.player_id
                ORDER BY pis.injury_prone_score DESC
                LIMIT 10
                """
                top_risk_players = pd.read_sql(top_risk_query, conn)
                
                # Get injury statistics by category
                injury_stats_query = """
                SELECT injury_category, COUNT(*) as count, AVG(days_out) as avg_days
                FROM injury_events
                GROUP BY injury_category
                ORDER BY count DESC
                """
                injury_stats = pd.read_sql(injury_stats_query, conn)
            
            # Create report
            report = {
                'integration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'injury_events_loaded': int(injury_events_count),
                'players_with_injury_history': int(injury_summaries_count),
                'top_injury_prone_players': top_risk_players.to_dict('records'),
                'injury_statistics_by_category': injury_stats.to_dict('records')
            }
            
            # Save report
            report_file = self.injuries_processed_path / 'integration_report.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Integration report saved to {report_file}")
            print(f"\nIntegration Summary:")
            print(f"   📊 Injury events loaded: {injury_events_count}")
            print(f"   👥 Players with injury history: {injury_summaries_count}")
            print(f"   🏆 Top injury-prone player: {top_risk_players.iloc[0]['player_name']} "
                  f"(Risk: {top_risk_players.iloc[0]['injury_prone_score']:.3f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return False
    
    def run_full_integration(self, force_refresh=False):
        """Run the complete integration pipeline"""
        print("=" * 60)
        print("STARTING INJURY DATA INTEGRATION PIPELINE")
        print("=" * 60)
        
        steps = [
            ("Setting up database tables", self.setup_injury_tables),
            ("Running R injury data scraping", lambda: self.run_r_scraping(force_refresh)),
            ("Processing injury data", self.process_injury_data),
        ]
        
        df_injuries = None
        
        for step_name, step_func in steps:
            print(f"\n📍 STEP: {step_name}")
            if step_name == "Processing injury data":
                df_injuries = step_func()
                if df_injuries is None:
                    print("❌ Integration failed at injury data processing")
                    return False
            else:
                success = step_func()
                if not success:
                    print(f"❌ Integration failed at {step_name}")
                    return False
        
        # Continue with data loading steps
        print(f"\n📍 STEP: Matching players to database")
        df_matched = self.match_players_to_database(df_injuries)
        if df_matched is None:
            print("❌ Integration failed at player matching")
            return False
        
        print(f"\n📍 STEP: Loading injury events to database")
        if not self.load_injury_events(df_matched):
            print("❌ Integration failed at loading injury events")
            return False
        
        print(f"\n📍 STEP: Calculating injury summaries")
        if not self.calculate_injury_summaries():
            print("❌ Integration failed at calculating summaries")
            return False
        
        print(f"\n📍 STEP: Generating integration report")
        if not self.generate_integration_report():
            print("❌ Integration failed at generating report")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 INJURY DATA INTEGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run injury analytics: psql -d your_db -f sql/injury_analytics_real.sql")
        print("2. Test ML model: python models/injury_predictor.py")
        print("3. Verify data: python python/test_queries.py")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Integrate injury data from R scraping to database')
    parser.add_argument('--force', action='store_true', help='Force re-scraping of injury data')
    parser.add_argument('--setup-only', action='store_true', help='Only setup database tables')
    parser.add_argument('--process-only', action='store_true', help='Only process existing R data')
    
    args = parser.parse_args()
    
    integrator = InjuryDataIntegrator()
    
    if args.setup_only:
        print("Setting up injury database tables only...")
        success = integrator.setup_injury_tables()
    elif args.process_only:
        print("Processing existing injury data only...")
        integrator.setup_injury_tables()
        df_injuries = integrator.process_injury_data()
        if df_injuries is not None:
            df_matched = integrator.match_players_to_database(df_injuries)
            if df_matched is not None:
                success = (integrator.load_injury_events(df_matched) and 
                          integrator.calculate_injury_summaries() and
                          integrator.generate_integration_report())
            else:
                success = False
        else:
            success = False
    else:
        success = integrator.run_full_integration(force_refresh=args.force)
    
    if success:
        print("\n✅ Integration completed successfully!")
    else:
        print("\n❌ Integration failed. Check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)