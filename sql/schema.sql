-- sql/schema.sql
-- Player Performance and Injury Prevention Database Schema
-- Most stats floats since they are stats per 90 minutes

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS injuries CASCADE;
DROP TABLE IF EXISTS training_load CASCADE;
DROP TABLE IF EXISTS player_performance CASCADE;
DROP TABLE IF EXISTS players CASCADE;

-- Create players table
CREATE TABLE players (
    player_id SERIAL PRIMARY KEY,
    player_name VARCHAR(100) NOT NULL,
    nation VARCHAR(50),
    year_born INTEGER,
    age INTEGER,
    position VARCHAR(10),
    team VARCHAR(50),
    league VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create player_performance table (detailed statistics)
CREATE TABLE player_performance (
    performance_id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(player_id),
    
    -- Basic game metrics
    minutes_played INTEGER,
    games_played INTEGER,
    games_started INTEGER,
    
    -- Carrying metrics
    carries DECIMAL(6,2),
    total_carry_distance DECIMAL(6,2),
    progressive_carry_distance DECIMAL(8,2),
    progressive_carries DECIMAL(6,2),
    carries_into_18 DECIMAL(6,2),
    dispossessed_carries DECIMAL(6,2),
    missed_carries DECIMAL(6,2),
    
    -- Duel metrics
    aerial_duels_won DECIMAL(6,2),
    aerial_duels_lost DECIMAL(6,2),
    aerial_duels_won_percentage DECIMAL(5,2),
    
    -- Passing metrics
    passes_completed DECIMAL(8,2),
    passes_attempted DECIMAL(8,2),
    pass_completion_percentage DECIMAL(5,2),
    total_passes_distance DECIMAL(10,2),
    progressive_pass_distance DECIMAL(10,2),
    progressive_passes_count DECIMAL(6,2),
    progressive_pass_received DECIMAL(6,2),
    
    -- Short passes
    short_passes_completed DECIMAL(8,2),
    short_passes_attempted DECIMAL(8,2),
    short_pass_completion_percentage DECIMAL(5,2),
    
    -- Medium passes
    medium_passes_completed DECIMAL(8,2),
    medium_passes_attempted DECIMAL(8,2),
    medium_pass_completion_percentage DECIMAL(5,2),
    
    -- Long passes
    long_passes_completed DECIMAL(8,2),
    long_passes_attempted DECIMAL(8,2),
    long_pass_completion_percentage DECIMAL(5,2),
    
    -- Shooting/creativity metrics
    shots DECIMAL(6,2),
    shots_on_target DECIMAL(6,2),
    shots_on_target_percentage DECIMAL(5,2),
    goals DECIMAL(6,2),
    goals_per_shot DECIMAL(5,3),
    goals_per_shot_on_target DECIMAL(5,3),
    assists DECIMAL(6,2),
    assisted_shots DECIMAL(6,2),
    completed_passes_into_18 DECIMAL(6,2),
    completed_crosses_into_18 DECIMAL(6,2),
    live_ball_passes DECIMAL(8,2),
    dead_ball_passes DECIMAL(6,2),
    passes_attempted_from_free_kicks DECIMAL(6,2),
    passes_offside DECIMAL(6,2),
    crosses DECIMAL(6,2),
    offsides DECIMAL(6,2),
    passes_blocked_by_opponent DECIMAL(6,2),

    -- Creating actions
    shot_creating_actions DECIMAL(6,2),
    shot_creating_actions_from_live_ball DECIMAL(6,2),
    shot_creating_actions_from_dead_ball DECIMAL(6,2),
    successful_dribbles_leading_to_shot DECIMAL(6,2),
    shots_leading_to_another_shot DECIMAL(6,2),
    fouls_drawn_leading_to_shot DECIMAL(6,2),
    defensive_actions_leading_to_shot DECIMAL(6,2),
    goal_creating_actions DECIMAL(6,2),
    live_ball_passes_leading_to_goal DECIMAL(6,2),
    dead_ball_passes_leading_to_goal DECIMAL(6,2),
    successful_dribbles_leading_to_goal DECIMAL(6,2),
    shots_leading_to_goal_scoring_shot DECIMAL(6,2),
    fouls_drawn_leading_to_goal DECIMAL(6,2),
    defensive_actions_leading_to_goal DECIMAL(6,2),
    
    -- Defensive metrics
    tackles DECIMAL(6,2),
    tackles_won DECIMAL(6,2),
    tackles_in_defensive_third DECIMAL(6,2),
    tackles_in_middle_third DECIMAL(6,2),
    tackles_in_attacking_third DECIMAL(6,2),
    number_of_dribblers_tackled DECIMAL(6,2),
    percentage_of_dribblers_tackled DECIMAL(6,2),
    number_of_times_dribbled_past_by_opponent DECIMAL(6,2),
    interceptions DECIMAL(6,2),
    number_of_tackles_and_interceptions DECIMAL(6,2),
    blocks DECIMAL(6,2),
    shots_blocked DECIMAL(6,2),
    passes_blocked DECIMAL(6,2),
    clearances DECIMAL(6,2),
    errors_leading_to_opponent_shot DECIMAL(6,2),
    
    -- Pressure and discipline
    pressures DECIMAL(6,2),
    pressure_success_rate DECIMAL(5,2),
    fouls_committed DECIMAL(6,2),
    fouls_drawn DECIMAL(6,2),
    yellow_cards DECIMAL(6,2),
    red_cards DECIMAL(6,2),
    second_yellow_card DECIMAL(6,2),
    
    -- Additional metrics
    touches DECIMAL(8,2),
    number_attempts_take_on_defender DECIMAL(6,2),
    number_defenders_taken_on_successfully DECIMAL(6,2),
    percentage_of_take_on_success DECIMAL(5,2),
    number_times_tackled_by_defender_during_take_on DECIMAL(6,2),
    percentage_tackled_by_defender_during_take_on DECIMAL(5,2),
    pass_received DECIMAL(8,2),
    penalty_kicks_won DECIMAL(6,2),
    penalty_kicks_conceded DECIMAL(6,2),
    penalty_kicks_attempted DECIMAL(6,2),
    own_goals DECIMAL(6,2),
    number_of_loose_balls_recovered DECIMAL(6,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create injury_events table for individual injury occurrences
CREATE TABLE injury_events (
    injury_id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(player_id),
    player_name VARCHAR(100),
    injury_date DATE,
    return_date DATE,
    days_out INTEGER,
    injury_type VARCHAR(200),        -- Detailed injury description
    injury_category VARCHAR(50),     -- Categorized: muscle, knee, ankle, etc.
    severity VARCHAR(20),            -- minor, moderate, severe, career-threatening
    league VARCHAR(50),
    team VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create player_injury_summary table for aggregated injury statistics
CREATE TABLE player_injury_summary (
    player_id INTEGER PRIMARY KEY REFERENCES players(player_id),
    total_injuries INTEGER DEFAULT 0,
    total_days_injured INTEGER DEFAULT 0,
    injury_prone_score DECIMAL(5,3) DEFAULT 0,    -- 0-1 risk score
    injury_frequency DECIMAL(5,2) DEFAULT 0,      -- injuries per year
    muscle_injuries INTEGER DEFAULT 0,
    knee_injuries INTEGER DEFAULT 0,
    ankle_injuries INTEGER DEFAULT 0,
    back_injuries INTEGER DEFAULT 0,
    last_injury_date DATE,
    avg_recovery_days DECIMAL(5,1),
    chronic_injury_flag BOOLEAN DEFAULT FALSE,    -- 3+ injuries
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- CREATE INDEXES FOR PERFORMANCE
-- ================================

-- Original performance indexes
CREATE INDEX idx_players_team ON players(team);
CREATE INDEX idx_players_name ON players(player_name);
CREATE INDEX idx_performance_player ON player_performance(player_id);

-- New injury table indexes
CREATE INDEX idx_injury_events_player ON injury_events(player_id);
CREATE INDEX idx_injury_events_date ON injury_events(injury_date);
CREATE INDEX idx_injury_events_category ON injury_events(injury_category);
CREATE INDEX idx_injury_summary_risk ON player_injury_summary(injury_prone_score DESC);
CREATE INDEX idx_injury_summary_chronic ON player_injury_summary(chronic_injury_flag);

-- ================================
-- CREATE MATERIALIZED VIEWS
-- ================================

-- Create a materialized view for quick risk analysis
CREATE MATERIALIZED VIEW player_risk_dashboard AS
SELECT 
    p.player_id,
    p.player_name,
    p.age,
    p.position,
    p.team,
    p.league,
    
    -- Performance metrics
    pp.minutes_played,
    pp.games_played,
    ROUND(pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0), 1) as avg_minutes_per_game,
    
    -- Injury history
    COALESCE(pis.total_injuries, 0) as career_injuries,
    COALESCE(pis.injury_prone_score, 0) as injury_prone_score,
    COALESCE(pis.chronic_injury_flag, FALSE) as chronic_injury_flag,
    pis.last_injury_date,
    
    -- Current injury status
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM injury_events ie 
            WHERE ie.player_id = p.player_id 
            AND ie.return_date > CURRENT_DATE
        ) THEN TRUE ELSE FALSE
    END as currently_injured,
    
    -- Recent injury count (last 6 months)
    (SELECT COUNT(*) FROM injury_events ie 
     WHERE ie.player_id = p.player_id 
     AND ie.injury_date >= CURRENT_DATE - INTERVAL '180 days') as recent_injuries,
    
    -- Risk calculation
    ROUND(
        -- Historical injury risk (40% weight)
        COALESCE(pis.injury_prone_score, 0) * 0.4 +
        
        -- Workload risk (20% weight)
        CASE 
            WHEN pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0) > 85 THEN 0.2
            WHEN pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0) > 75 THEN 0.1
            ELSE 0.05
        END +
        
        -- Age risk (15% weight)
        CASE 
            WHEN p.age > 32 THEN 0.15
            WHEN p.age > 29 THEN 0.10
            WHEN p.age > 26 THEN 0.05
            ELSE 0.02
        END +
        
        -- Recent injury risk (25% weight)
        CASE 
            WHEN pis.last_injury_date > CURRENT_DATE - INTERVAL '60 days' THEN 0.25
            WHEN pis.last_injury_date > CURRENT_DATE - INTERVAL '120 days' THEN 0.15
            WHEN pis.last_injury_date > CURRENT_DATE - INTERVAL '180 days' THEN 0.10
            ELSE 0
        END,
        3
    ) AS current_risk_score,
    
    CURRENT_TIMESTAMP as last_updated
    
FROM players p
JOIN player_performance pp ON p.player_id = pp.player_id
LEFT JOIN player_injury_summary pis ON p.player_id = pis.player_id
WHERE pp.minutes_played > 360;  -- Minimum playing time

-- Create index on the materialized view
CREATE INDEX idx_risk_dashboard_risk ON player_risk_dashboard(current_risk_score DESC);
CREATE INDEX idx_risk_dashboard_team ON player_risk_dashboard(team);
CREATE INDEX idx_risk_dashboard_currently_injured ON player_risk_dashboard(currently_injured);

-- ================================
-- CREATE HELPFUL FUNCTIONS
-- ================================

-- Function to refresh the risk dashboard
CREATE OR REPLACE FUNCTION refresh_risk_dashboard()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW player_risk_dashboard;
END;
$$ LANGUAGE plpgsql;

-- Function to get player injury risk level
CREATE OR REPLACE FUNCTION get_risk_level(risk_score DECIMAL)
RETURNS VARCHAR(20) AS $$
BEGIN
    CASE 
        WHEN risk_score >= 0.7 THEN RETURN 'CRITICAL';
        WHEN risk_score >= 0.5 THEN RETURN 'HIGH';
        WHEN risk_score >= 0.3 THEN RETURN 'MODERATE';
        ELSE RETURN 'LOW';
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- ================================
-- CREATE SAMPLE QUERIES VIEW
-- ================================

-- Create a view with common injury queries for easy access
CREATE VIEW injury_quick_stats AS
SELECT 
    'Total Players' as metric,
    COUNT(*)::TEXT as value
FROM players
UNION ALL
SELECT 
    'Players with Injury History',
    COUNT(*)::TEXT
FROM player_injury_summary
WHERE total_injuries > 0
UNION ALL
SELECT 
    'Currently Injured',
    COUNT(*)::TEXT
FROM player_risk_dashboard
WHERE currently_injured = TRUE
UNION ALL
SELECT 
    'High Risk Players',
    COUNT(*)::TEXT
FROM player_risk_dashboard
WHERE current_risk_score >= 0.5
UNION ALL
SELECT 
    'Critical Risk Players',
    COUNT(*)::TEXT
FROM player_risk_dashboard
WHERE current_risk_score >= 0.7;

-- Print schema creation summary
DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'INJURY PREVENTION DATABASE SCHEMA CREATED';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Core Tables:';
    RAISE NOTICE '  ✓ players (player information)';
    RAISE NOTICE '  ✓ player_performance (stats & metrics)';
    RAISE NOTICE '  ✓ injury_events (individual injuries)';
    RAISE NOTICE '  ✓ player_injury_summary (aggregated stats)';
    RAISE NOTICE '';
    RAISE NOTICE 'Views & Functions:';
    RAISE NOTICE '  ✓ player_risk_dashboard (materialized view)';
    RAISE NOTICE '  ✓ injury_quick_stats (summary view)';
    RAISE NOTICE '  ✓ refresh_risk_dashboard() function';
    RAISE NOTICE '  ✓ get_risk_level() function';
    RAISE NOTICE '';
    RAISE NOTICE 'Next Steps:';
    RAISE NOTICE '  1. Load player data: python python/data_collector.py';
    RAISE NOTICE '  2. Integrate injuries: python python/injury_integrator.py';
    RAISE NOTICE '  3. Run analytics: psql -f sql/injury_analytics_real.sql';
    RAISE NOTICE '============================================';
END $$;