-- sql/schema.sql
-- Player Performance and Injury Prevention Database Schema

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS injuries CASCADE;
DROP TABLE IF EXISTS training_load CASCADE;
DROP TABLE IF EXISTS player_performance CASCADE;
DROP TABLE IF EXISTS players CASCADE;

-- Create players table
CREATE TABLE players (
    player_id SERIAL PRIMARY KEY,
    player_name VARCHAR(100) NOT NULL,
    age INTEGER,
    position VARCHAR(20),
    team VARCHAR(100),
    league VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create player_performance table (detailed statistics)
CREATE TABLE player_performance (
    performance_id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(player_id),
    season VARCHAR(9),
    
    -- Basic game metrics
    minutes_played INTEGER,
    games_played INTEGER,
    games_started INTEGER,
    
    -- Carrying metrics
    carries INTEGER,
    total_carry_distance DECIMAL(10,2),
    progressive_carry_distance DECIMAL(10,2),
    progressive_carries INTEGER,
    carries_into_18 INTEGER,
    dispossessed_carries INTEGER,
    missed_carries INTEGER,
    
    -- Duel metrics
    aerial_duels_won INTEGER,
    aerial_duels_lost INTEGER,
    aerial_duels_won_percentage DECIMAL(5,2),
    
    -- Passing metrics
    passes_completed INTEGER,
    passes_attempted INTEGER,
    pass_completion_percentage DECIMAL(5,2),
    total_passes_distance DECIMAL(10,2),
    progressive_pass_distance DECIMAL(10,2),
    progressive_passes_count INTEGER,
    progressive_pass_received INTEGER,
    
    -- Short passes
    short_passes_completed INTEGER,
    short_passes_attempted INTEGER,
    short_pass_completion_percentage DECIMAL(5,2),
    
    -- Medium passes
    medium_passes_completed INTEGER,
    medium_passes_attempted INTEGER,
    medium_pass_completion_percentage DECIMAL(5,2),
    
    -- Long passes
    long_passes_completed INTEGER,
    long_passes_attempted INTEGER,
    long_pass_completion_percentage DECIMAL(5,2),
    
    -- Shooting/creativity metrics
    shots INTEGER,
    shots_on_target INTEGER,
    shots_on_target_percentage DECIMAL(5,2),
    goals INTEGER,
    goals_per_shot DECIMAL(5,2),
    goals_per_shot_on_target DECIMAL(5,2),
    assists INTEGER,
    assisted_shots INTEGER,
    completed_passes_into_18 INTEGER,
    completed_crosses_into_18 INTEGER,
    live_ball_passes INTEGER,
    dead_ball_passes INTEGER,
    passes_attempted_from_free_kicks INTEGER,
    passes_offside INTEGER,
    crosses INTEGER,
    offsides INTEGER,
    passes_blocked_by_opponent INTEGER,

    -- Creating actions
    shot_creating_actions INTEGER,
    shot_creating_actions_from_live_ball INTEGER,
    shot_creating_actions_from_dead_ball INTEGER,
    successful_dribbles_leading_to_shot INTEGER,
    shots_leading_to_another_shot INTEGER,
    fouls_drawn_leading_to_shot INTEGER,
    defensive_actions_leading_to_shot INTEGER,
    goal_creating_actions INTEGER,
    live_ball_passes_leading_to_goal INTEGER,
    dead_ball_passes_leading_to_goal INTEGER,
    successful_dribbles_leading_to_goal INTEGER,
    shots_leading_to_goal_scoring_shot INTEGER,
    fouls_drawn_leading_to_goal INTEGER,
    defensive_actions_leading_to_goal INTEGER,
    
    -- Defensive metrics
    tackles INTEGER,
    tackles_won INTEGER,
    tackles_in_defensive_third INTEGER,
    tackles_in_middle_third INTEGER,
    tackles_in_attacking_third INTEGER,
    number_of_dribblers_tackled INTEGER,
    percentage_of_dribblers_tackled DECIMAL(5,2),
    number_of_times_dribbled_past_by_opponent INTEGER,
    interceptions INTEGER,
    number_of_tackles_and_interceptions INTEGER,
    blocks INTEGER,
    shots_blocked INTEGER,
    passes_blocked INTEGER,
    clearances INTEGER,
    errors_leading_to_opponent_shot INTEGER,
    
    -- Pressure and discipline
    pressures INTEGER,
    pressure_success_rate DECIMAL(5,2),
    fouls_committed INTEGER,
    fouls_drawn INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    second_yellow_card INTEGER,
    
    -- Additional metrics
    touches INTEGER,
    number_attempts_take_on_defender INTEGER,
    number_defenders_taken_on_successfully INTEGER,
    percentage_of_take_on_success DECIMAL(5,2),
    number_times_tackled_by_defender_during_take_on INTEGER,
    percentage_tackled_by_defender_during_take_on DECIMAL(5,2),
    pass_received INTEGER,
    penalty_kicks_won INTEGER,
    penalty_kicks_conceded INTEGER,
    penalty_kicks_attempted INTEGER,
    own_goals INTEGER,
    number_of_loose_balls_recovered INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_players_team ON players(team);
CREATE INDEX idx_players_name ON players(player_name);
CREATE INDEX idx_performance_player ON player_performance(player_id);
CREATE INDEX idx_performance_season ON player_performance(season);