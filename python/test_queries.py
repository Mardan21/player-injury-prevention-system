"""
Test SQL queries and save results to outputs directory
"""

import pandas as pd
from sqlalchemy import text
from db_config import engine
import os
from datetime import datetime

class QueryTester:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sql_path = os.path.join(self.base_path, 'sql')
        self.output_path = os.path.join(self.base_path, 'outputs')
        
        # Create outputs directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
    def test_workload_query(self):
        """Test the workload analysis query"""
        print("\n=== Testing Workload Analysis Query ===")
        
        test_query = """
        WITH player_workload AS (
            SELECT 
                p.player_id,
                p.player_name,
                p.position,
                p.team,
                pp.goals,
                pp.assists,
                pp.games_played
            FROM players p
            JOIN player_performance pp ON p.player_id = pp.player_id
            WHERE pp.minutes_played > 360  -- Minimum 360 minutes played
            AND pp.games_played >= 5       -- Minimum 5 games played
        )
        SELECT * FROM player_workload
        ORDER BY assists DESC
        LIMIT 30
        """

        query="""
        WITH player_workload AS (
            SELECT 
                p.player_id,
                p.player_name,
                p.position,
                p.team,
                p.age,
                pp.minutes_played,
                pp.games_played,
                pp.fouls_committed * 0.5 + pp.yellow_cards * 1.0 + pp.red_cards * 3.0 AS disciplinary_score,

                -- High intensity actions
                pp.progressive_carries + pp.progressive_passes_count AS progressive_actions_per_90,
                pp.tackles_won + pp.number_of_dribblers_tackled AS high_intensity_defensive_actions_per_90,
                (pp.progressive_carries * 0.25 +
                  pp.progressive_carry_distance * 0.25 +
                  pp.successful_dribbles_leading_to_shot * 0.2 +
                  pp.number_attempts_take_on_defender * 0.15 +
                  (pp.aerial_duels_won + pp.aerial_duels_lost) * 0.15 
                ) AS high_intensity_score,

                -- Calculate workload intensity
                (pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0)) AS avg_minutes_per_game,                
                ((pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0)) * 0.4 +
                  (pp.games_played * 1.0 / 295) * (pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0)) * 0.2 +
                  (pp.progressive_carries * 0.25 +
                    pp.progressive_carry_distance * 0.25 +
                    pp.successful_dribbles_leading_to_shot * 0.2 +
                    pp.number_attempts_take_on_defender * 0.15 +
                    (pp.aerial_duels_won + pp.aerial_duels_lost) * 0.15
                  ) * 0.25 +
                  (pp.tackles +
                    pp.interceptions +
                    pp.clearances +
                    pp.blocks
                  ) * 0.15
                ) AS fatigue_score,

                -- Defensive load indicators
                (pp.tackles_won * 0.5 +
                  pp.interceptions * 0.5 +
                  pp.clearances * 0.1 + 
                  pp.number_of_dribblers_tackled * 0.5 +
                  pp.shots_blocked * 0.1 +
                  pp.passes_blocked * 0.1 +
                  pp.aerial_duels_won * 0.25 + 
                  pp.number_of_loose_balls_recovered * 0.1 
                ) AS dap90,

                -- Creative load indicators
                (pp.assists * 0.7 +
                  pp.assisted_shots * 0.5 +
                  pp.progressive_passes_count * 0.1 +
                  pp.progressive_carries + 0.05 +
                  pp.pass_completion_percentage * 0.1 +
                  pp.completed_crosses_into_18 * 0.01     +
                  pp.shot_creating_actions * 0.25 +
                  pp.successful_dribbles_leading_to_shot * 0.25           
                ) AS cap90,

                -- Offensive load indicators
                (pp.assists * 1.0 + 
                  pp.shots_on_target * 0.5 + 
                  pp.goals_per_shot_on_target * 0.8 +
                  pp.shots_leading_to_another_shot * 0.1 +
                  pp.successful_dribbles_leading_to_goal * 0.2 +
                  pp.shots_leading_to_goal_scoring_shot * 0.5
                ) AS oap90
              
            FROM players p
            JOIN player_performance pp ON p.player_id = pp.player_id
            WHERE pp.minutes_played > 500  -- Minimum 500 minutes played
            AND pp.games_played >= 5       -- Minimum 5 games played
        ),
        workload_percentiles AS (
            SELECT
                *,
                -- Calculate composite workload score
                (avg_minutes_per_game * 0.4 +
                dap90 * 0.4 + 
                oap90 * 0.3 +
                cap90 * 0.25) AS composite_workload_score
            FROM player_workload
        ),
        workload_percentiles_with_ranks AS (
            SELECT
                *,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY avg_minutes_per_game) AS minutes_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER by fatigue_score) AS fatigue_score_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY disciplinary_score) AS disciplinary_score_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY oap90) AS oap90_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY cap90) AS cap90_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY dap90) AS dap90_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY high_intensity_score) AS high_intensity_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY composite_workload_score) AS composite_workload_score_percentile,

                -- Injury risk index
                (fatigue_score * 0.4 +
                  disciplinary_score * 0.2 +
                  age * 0.1 
                ) AS injury_risk_index
            FROM workload_percentiles
        )
        SELECT 
            player_id,
            player_name,
            position,
            team,
            avg_minutes_per_game,
            minutes_percentile,
            dap90,
            dap90_percentile,
            oap90,
            oap90_percentile,
            cap90,
            cap90_percentile,
            high_intensity_score,
            high_intensity_percentile,
            fatigue_score,
            fatigue_score_percentile,
            injury_risk_index,
            composite_workload_score,
            composite_workload_score_percentile,
            CASE
                WHEN minutes_percentile > 0.9 AND (dap90_percentile > 0.9 OR oap90_percentile > 0.9 OR cap90_percentile > 0.9) THEN '1. VERY HIGH'
                WHEN minutes_percentile > 0.7 AND (dap90_percentile > 0.8 OR oap90_percentile > 0.8 OR cap90_percentile > 0.8) THEN '2. HIGH'
                WHEN minutes_percentile > 0.5 THEN '3. MODERATE'
                ELSE '4. LOW'
            END AS workload_risk_level
        FROM workload_percentiles_with_ranks
        ORDER BY composite_workload_score_percentile DESC
        """
        
        try:
            df = pd.read_sql(query, engine)
            print(f"Query successful! Retrieved {len(df)} rows")
            print("\nTop Players by offensive and defensive load:")
            print(df[['player_name', 'fatigue_score_percentile', 'injury_risk_index', 'high_intensity_percentile', 'composite_workload_score_percentile', 'workload_risk_level']].head(50))
            
            # Save to CSV
            output_file = os.path.join(self.output_path, 'test_analysis.csv')
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
    
    def test_injury_risk_query(self):
        """Test the injury risk prediction query"""
        print("\n=== Testing Injury Risk Prediction Query ===")
        
        query = """
        WITH player_risk_factors AS (
            SELECT 
                p.player_id,
                p.player_name,
                p.age,
                p.position,
                p.team,

                -- Age risk factor (normalized 0-1)
                CASE 
                    WHEN p.age < 20 THEN 0.35
                    WHEN p.age <= 23 THEN 0.31
                    WHEN p.age <= 27 THEN 0.27
                    WHEN p.age <= 30 THEN 0.31
                    WHEN p.age <= 33 THEN 0.35
                    ELSE 0.4
                END AS age_risk_factor,
                
                -- Workload risk factor
                pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0) AS avg_minutes,

                -- Physical stress indicators
                pp.fouls_committed + pp.fouls_drawn AS total_fouls_involved,
                pp.tackles + pp.number_of_times_dribbled_past_by_opponent AS tackle_exposure,
                pp.aerial_duels_won + pp.aerial_duels_lost AS aerial_exposure,

                -- Fatigue indicators
                pp.errors_leading_to_opponent_shot AS fatigue_errors,
                pp.dispossessed_carries + pp.missed_carries AS ball_control_errors,

                -- High intensity actions (from workload query)
                pp.progressive_carries + pp.progressive_passes_count AS progressive_actions_per_90,
                pp.tackles_won + pp.number_of_dribblers_tackled AS high_intensity_defensive_actions_per_90,
                
                -- High intensity score
                (pp.progressive_carries * 0.25 +
                pp.progressive_carry_distance * 0.25 +
                pp.successful_dribbles_leading_to_shot * 0.2 +
                pp.number_attempts_take_on_defender * 0.15 +
                (pp.aerial_duels_won + pp.aerial_duels_lost) * 0.15 
                ) AS high_intensity_score,

                -- Fatigue score components
                (pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0)) AS avg_minutes_per_game,
                (pp.carries * 0.7 + 
                pp.total_carry_distance * 1.0 + 
                pp.aerial_duels_won * 0.5 + 
                pp.progressive_carries * 1.2 + 
                pp.progressive_passes_count * 0.7 + 
                pp.shots * 0.5 + 
                pp.passes_attempted * 0.3 + 
                pp.number_of_loose_balls_recovered * 0.5 + 
                pp.tackles_won * 0.5) AS fatigue_per_90,

                -- Recovery demand score
                (
                    pp.tackles * 2 + 
                    pp.fouls_committed * 1.5 + 
                    pp.aerial_duels_won * 1.2 +                 -- High-impact landing
                    pp.aerial_duels_lost * 1.2 +                -- High-impact landing
                    pp.progressive_carries * 1.5 +              -- High-intensity-movement
                    pp.number_attempts_take_on_defender * 1.3 + -- Explosive movement
                    pp.shots_blocked * 1.5
                ) AS recovery_demand

            FROM players p
            JOIN player_performance pp ON p.player_id = pp.player_id
            WHERE pp.minutes_played > 360  -- Minimum 360 minutes played
            AND pp.games_played >= 5       -- Minimum 5 games played
        ),
        risk_percentiles AS (
            SELECT 
                *,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY avg_minutes) AS workload_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY total_fouls_involved) AS foul_involvement_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY tackle_exposure) AS tackle_risk_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY recovery_demand) AS recovery_demand_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY ball_control_errors) AS error_rate_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY high_intensity_score) AS high_intensity_percentile,
                PERCENT_RANK() OVER (PARTITION BY position ORDER BY fatigue_per_90) AS fatigue_percentile
            FROM player_risk_factors
        ),
        injury_risk_scores AS (
            SELECT 
                player_id,
                player_name,
                age,
                position,
                team,
                
                -- Calculate weighted injury risk score with new components
                (age_risk_factor * 0.15 +                   -- Reduced age weight
                workload_percentile * 0.20 +                -- Reduced workload weight
                foul_involvement_percentile * 0.10 +        -- Reduced contact weight
                tackle_risk_percentile * 0.15 +             -- Reduced tackle weight
                recovery_demand_percentile * 0.10 +         -- Kept same
                error_rate_percentile * 0.10 +              -- Kept same
                high_intensity_percentile * 0.10 +          -- New component
                fatigue_percentile * 0.10                   -- New component
                ) AS injury_risk_score,
                
                -- Individual risk components
                age_risk_factor AS age_risk,
                workload_percentile AS workload_risk,
                foul_involvement_percentile AS contact_risk,
                tackle_risk_percentile AS tackle_risk,
                recovery_demand_percentile AS recovery_risk,
                high_intensity_percentile AS intensity_risk,
                fatigue_percentile AS fatigue_risk
                
            FROM risk_percentiles
        )
        SELECT 
            *,
            CASE 
                WHEN injury_risk_score >= 0.8 THEN 'CRITICAL'   -- Top 20% risk
                WHEN injury_risk_score >= 0.65 THEN 'VERY HIGH' -- Next 15% risk
                WHEN injury_risk_score >= 0.5 THEN 'HIGH'       -- Next 15% risk
                WHEN injury_risk_score >= 0.35 THEN 'MODERATE'  -- Next 15% risk
                ELSE 'LOW'                                      -- Bottom 35% risk
            END AS risk_category,
            RANK() OVER (ORDER BY injury_risk_score DESC) AS risk_rank
        FROM injury_risk_scores
        ORDER BY injury_risk_score DESC
        LIMIT 50
        """
        
        try:
            df = pd.read_sql(text(query), engine)
            print(f"Query successful! Retrieved {len(df)} rows")
            print("\nTop 50 highest risk players:")
            print(df[['player_name', 'injury_risk_score', 'risk_category', 'intensity_risk', 'fatigue_risk']].head(50))
            
            # Save to CSV
            output_file = os.path.join(self.output_path, 'injury_risk_analysis.csv')
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
    
    def test_team_analysis_query(self):
        """Test the team workload distribution query"""
        print("\n=== Testing Team Workload Distribution Query ===")
        
        query = """
        WITH team_workload_stats AS (
            SELECT 
                p.team,
                COUNT(DISTINCT p.player_id) AS total_players,
                AVG(pp.minutes_played) AS avg_team_minutes,
                STDDEV(pp.minutes_played) AS stddev_team_minutes,
                MAX(pp.minutes_played) AS max_player_minutes,
                MIN(pp.minutes_played) AS min_player_minutes,
                
                -- Gini coefficient for workload inequality
                AVG(pp.minutes_played) - 
                (SELECT AVG(ABS(pp1.minutes_played - pp2.minutes_played))
                FROM player_performance pp1
                JOIN players p1 ON pp1.player_id = p1.player_id
                JOIN player_performance pp2 ON p1.team = p.team
                JOIN players p2 ON pp2.player_id = p2.player_id
                WHERE p1.team = p.team) / (2 * AVG(pp.minutes_played)) AS workload_balance_score
                
            FROM players p
            JOIN player_performance pp ON p.player_id = pp.player_id
            GROUP BY p.team
        ),
        team_risk_analysis AS (
            SELECT 
                team,
                total_players,
                avg_team_minutes AS avg_minutes,
                stddev_team_minutes AS minutes_variance,
                max_player_minutes - min_player_minutes AS minutes_range,
                stddev_team_minutes / NULLIF(avg_team_minutes, 0) * 100 AS coefficient_of_variation,
                
                -- Risk categorization
                CASE 
                    WHEN stddev_team_minutes / NULLIF(avg_team_minutes, 0) > 0.5 THEN 'HIGH'
                    WHEN stddev_team_minutes / NULLIF(avg_team_minutes, 0) > 0.3 THEN 'MODERATE'
                    ELSE 'LOW'
                END AS workload_distribution_risk
                
            FROM team_workload_stats
        )
        SELECT 
            t.*,
            
            -- Top 3 most used players per team
            (SELECT STRING_AGG(player_name || ' (' || minutes_played || ')', ', ' ORDER BY minutes_played DESC)
            FROM (
                SELECT p.player_name, pp.minutes_played
                FROM players p
                JOIN player_performance pp ON p.player_id = pp.player_id
                WHERE p.team = t.team
                ORDER BY pp.minutes_played DESC
                LIMIT 3
            ) top_players) AS top_3_workload_players
            
        FROM team_risk_analysis t
        ORDER BY coefficient_of_variation DESC
        LIMIT 50
        """
        
        try:
            df = pd.read_sql(text(query), engine)
            print(f"Query successful! Retrieved {len(df)} rows")
            print("\nTeams with highest workload imbalance:")
            print(df[['team', 'total_players', 'coefficient_of_variation', 'workload_distribution_risk']].head())
            
            # Save to CSV
            output_file = os.path.join(self.output_path, 'team_workload_distribution.csv')
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
    
    def create_materialized_view(self):
        """Create the materialized view for performance"""
        print("\n=== Creating Materialized View ===")
        
        try:
            with engine.connect() as conn:
                # Drop if exists
                conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS player_risk_summary CASCADE"))
                conn.commit()
                
                # Create materialized view
                create_view_query = """
                CREATE MATERIALIZED VIEW player_risk_summary AS
                WITH latest_performance AS (
                    SELECT 
                        p.player_id,
                        p.player_name,
                        p.age,
                        p.position,
                        p.team,
                        pp.minutes_played,
                        pp.games_played,
                        pp.tackles + pp.interceptions AS defensive_load,
                        pp.carries + pp.passes_attempted AS offensive_load,
                        pp.fouls_committed + pp.fouls_drawn AS contact_exposure,
                        pp.yellow_cards + pp.red_cards * 2 AS discipline_issues,
                        pp.errors_leading_to_opponent_shot + pp.dispossessed_carries AS fatigue_indicators
                    FROM players p
                    JOIN player_performance pp ON p.player_id = pp.player_id
                )
                SELECT 
                    player_id,
                    player_name,
                    age,
                    position,
                    team,
                    minutes_played,
                    games_played,
                    minutes_played * 1.0 / NULLIF(games_played, 0) AS avg_minutes_per_game,
                    defensive_load,
                    offensive_load,
                    contact_exposure,
                    discipline_issues,
                    fatigue_indicators,
                    
                    -- Simple risk score for quick reference
                    
                    CASE WHEN age > 30 THEN 0.3 ELSE 0.1 END +
                    CASE WHEN minutes_played * 1.0 / NULLIF(games_played, 0) > 80 THEN 0.3 ELSE 0.1 END +
                    CASE WHEN contact_exposure > 50 THEN 0.2 ELSE 0.1 END +
                    CASE WHEN fatigue_indicators > 10 THEN 0.2 ELSE 0.1 END AS quick_risk_score,
                    
                    CURRENT_TIMESTAMP AS last_updated
                    
                FROM latest_performance
                """
                
                conn.execute(text(create_view_query))
                conn.commit()
                
                # Create indexes
                conn.execute(text("CREATE INDEX idx_risk_summary_player ON player_risk_summary(player_id)"))
                conn.execute(text("CREATE INDEX idx_risk_summary_team ON player_risk_summary(team)"))
                conn.execute(text("CREATE INDEX idx_risk_summary_risk ON player_risk_summary(quick_risk_score DESC)"))
                conn.commit()
                
                print("Materialized view created successfully!")
                
                # Test the view
                result = conn.execute(text("SELECT COUNT(*) FROM player_risk_summary"))
                count = result.scalar()
                print(f"View contains {count} records")

            return True
                
        except Exception as e:
            print(f"Error creating materialized view: {e}")
            return False
    
    def generate_summary_report(self):
        """Generate a summary report of all analyses"""
        print("\n=== Generating Summary Report ===")
        
        summary = {
            'Report Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Database Stats': {},
            'Risk Analysis': {},
            'Recommendations': []
        }
        
        try:
            with engine.connect() as conn:
                # Get database stats
                player_count = conn.execute(text("SELECT COUNT(*) FROM players")).scalar()
                performance_count = conn.execute(text("SELECT COUNT(*) FROM player_performance")).scalar()
                
                summary['Database Stats'] = {
                    'Total Players': player_count,
                    'Performance Records': performance_count
                }
                
                # Get risk stats from materialized view
                risk_query = """
                SELECT 
                    COUNT(*) FILTER (WHERE quick_risk_score >= 0.7) AS critical_risk,
                    COUNT(*) FILTER (WHERE quick_risk_score >= 0.5 AND quick_risk_score < 0.7) AS high_risk,
                    COUNT(*) FILTER (WHERE quick_risk_score >= 0.3 AND quick_risk_score < 0.5) AS moderate_risk,
                    COUNT(*) FILTER (WHERE quick_risk_score < 0.3) AS low_risk
                FROM player_risk_summary
                """
                
                risk_result = conn.execute(text(risk_query)).fetchone()
                summary['Risk Analysis'] = {
                    'Critical Risk Players': risk_result[0],
                    'High Risk Players': risk_result[1],
                    'Moderate Risk Players': risk_result[2],
                    'Low Risk Players': risk_result[3]
                }
                
                # Generate recommendations
                summary['Recommendations'] = [
                    f"Monitor {risk_result[0]} players in critical risk category immediately",
                    f"Schedule recovery sessions for {risk_result[1]} high-risk players",
                    "Review training intensity for teams with high workload variance",
                    "Implement rotation policy for players over 30 with high minutes"
                ]
                
            # Save summary
            import json
            output_file = os.path.join(self.output_path, 'analysis_summary.json')
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print("\nSummary Report:")
            print(json.dumps(summary, indent=2))
            print(f"\nSummary saved to: {output_file}")

            return True
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return False

if __name__ == "__main__":
    tester = QueryTester()
    
    # Run all tests
    print("Starting SQL query tests...")
    
    # Test individual queries
    tester.test_workload_query()
    tester.test_injury_risk_query()
    tester.test_team_analysis_query()
    
    # Create materialized view
    tester.create_materialized_view()
    
    # Generate summary report
    tester.generate_summary_report()
    
    print("\n=== All tests completed! ===")