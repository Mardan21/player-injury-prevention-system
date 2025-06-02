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
    -- season VARCHAR(9),
    
    -- Basic game metrics
    minutes_played INTEGER,
    games_played INTEGER,
    games_started INTEGER,
    
    -- Carrying metrics
    carries DECIMAL(4,2),
    total_carry_distance DECIMAL(4,2),
    progressive_carry_distance DECIMAL(4,2),
    progressive_carries DECIMAL(4,2),
    carries_into_18 DECIMAL(4,2),
    dispossessed_carries DECIMAL(4,2),
    missed_carries DECIMAL(4,2),
    
    -- Duel metrics
    aerial_duels_won DECIMAL(4,2),
    aerial_duels_lost DECIMAL(4,2),
    aerial_duels_won_percentage DECIMAL(4,2),
    
    -- Passing metrics
    passes_completed DECIMAL(4,2),
    passes_attempted DECIMAL(4,2),
    pass_completion_percentage DECIMAL(4,2),
    total_passes_distance DECIMAL(4,2),
    progressive_pass_distance DECIMAL(4,2),
    progressive_passes_count DECIMAL(4,2),
    progressive_pass_received DECIMAL(4,2),
    
    -- Short passes
    short_passes_completed DECIMAL(4,2),
    short_passes_attempted DECIMAL(4,2),
    short_pass_completion_percentage DECIMAL(4,2),
    
    -- Medium passes
    medium_passes_completed DECIMAL(4,2),
    medium_passes_attempted DECIMAL(4,2),
    medium_pass_completion_percentage DECIMAL(4,2),
    
    -- Long passes
    long_passes_completed DECIMAL(4,2),
    long_passes_attempted DECIMAL(4,2),
    long_pass_completion_percentage DECIMAL(4,2),
    
    -- Shooting/creativity metrics
    shots DECIMAL(4,2),
    shots_on_target DECIMAL(4,2),
    shots_on_target_percentage DECIMAL(4,2),
    goals DECIMAL(4,2),
    goals_per_shot DECIMAL(4,2),
    goals_per_shot_on_target DECIMAL(4,2),
    assists DECIMAL(4,2),
    assisted_shots DECIMAL(4,2),
    completed_passes_into_18 DECIMAL(4,2),
    completed_crosses_into_18 DECIMAL(4,2),
    live_ball_passes DECIMAL(4,2),
    dead_ball_passes DECIMAL(4,2),
    passes_attempted_from_free_kicks DECIMAL(4,2),
    passes_offside DECIMAL(4,2),
    crosses DECIMAL(4,2),
    offsides DECIMAL(4,2),
    passes_blocked_by_opponent DECIMAL(4,2),

    -- Creating actions
    shot_creating_actions DECIMAL(4,2),
    shot_creating_actions_from_live_ball DECIMAL(4,2),
    shot_creating_actions_from_dead_ball DECIMAL(4,2),
    successful_dribbles_leading_to_shot DECIMAL(4,2),
    shots_leading_to_another_shot DECIMAL(4,2),
    fouls_drawn_leading_to_shot DECIMAL(4,2),
    defensive_actions_leading_to_shot DECIMAL(4,2),
    goal_creating_actions DECIMAL(4,2),
    live_ball_passes_leading_to_goal DECIMAL(4,2),
    dead_ball_passes_leading_to_goal DECIMAL(4,2),
    successful_dribbles_leading_to_goal DECIMAL(4,2),
    shots_leading_to_goal_scoring_shot DECIMAL(4,2),
    fouls_drawn_leading_to_goal DECIMAL(4,2),
    defensive_actions_leading_to_goal DECIMAL(4,2),
    
    -- Defensive metrics
    tackles DECIMAL(4,2),
    tackles_won DECIMAL(4,2),
    tackles_in_defensive_third DECIMAL(4,2),
    tackles_in_middle_third DECIMAL(4,2),
    tackles_in_attacking_third DECIMAL(4,2),
    number_of_dribblers_tackled DECIMAL(4,2),
    percentage_of_dribblers_tackled DECIMAL(4,2),
    number_of_times_dribbled_past_by_opponent DECIMAL(4,2),
    interceptions DECIMAL(4,2),
    number_of_tackles_and_interceptions DECIMAL(4,2),
    blocks DECIMAL(4,2),
    shots_blocked DECIMAL(4,2),
    passes_blocked DECIMAL(4,2),
    clearances DECIMAL(4,2),
    errors_leading_to_opponent_shot DECIMAL(4,2),
    
    -- Pressure and discipline
    pressures DECIMAL(4,2),
    pressure_success_rate DECIMAL(4,2),
    fouls_committed DECIMAL(4,2),
    fouls_drawn DECIMAL(4,2),
    yellow_cards DECIMAL(4,2),
    red_cards DECIMAL(4,2),
    second_yellow_card DECIMAL(4,2),
    
    -- Additional metrics
    touches DECIMAL(4,2),
    number_attempts_take_on_defender DECIMAL(4,2),
    number_defenders_taken_on_successfully DECIMAL(4,2),
    percentage_of_take_on_success DECIMAL(4,2),
    number_times_tackled_by_defender_during_take_on DECIMAL(4,2),
    percentage_tackled_by_defender_during_take_on DECIMAL(4,2),
    pass_received DECIMAL(4,2),
    penalty_kicks_won DECIMAL(4,2),
    penalty_kicks_conceded DECIMAL(4,2),
    penalty_kicks_attempted DECIMAL(4,2),
    own_goals DECIMAL(4,2),
    number_of_loose_balls_recovered DECIMAL(4,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_players_team ON players(team);
CREATE INDEX idx_players_name ON players(player_name);
CREATE INDEX idx_performance_player ON player_performance(player_id);
CREATE INDEX idx_performance_season ON player_performance(season);