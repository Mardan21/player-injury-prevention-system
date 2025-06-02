-- workload_analysis.sql
-- Complex SQL queries for player workload and injury risk analysis
-- Uses window functions, CTEs, and advanced analytics

-- 1. Player Workload Intensity Score
-- Calculates rolling averages and identifies workload spikes
WITH player_workload AS (
    SELECT 
        p.player_id,
        p.player_name,
        p.position,
        p.team,
        pp.minutes_played,
        pp.games_played,

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
          pp.completed_crosses_into_18 * 0.01 +
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
        ) AS oap90,
        
        pp.fouls_committed * 0.5 + pp.yellow_cards * 1.0 + pp.red_cards * 3.0 AS discipline_score,

        -- Calculate workload intensity
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

        -- Calculate workload intensity
        (pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0)) AS avg_minutes_per_game,

        -- High intensity actions
        pp.progressive_carries + pp.progressive_passes_count AS progressive_actions_per_90,
        pp.tackles_won + pp.number_of_dribblers_tackled AS high_intensity_defensive_actions_per_90,
        (pp.progressive_carries * 0.25 +
            pp.progressive_carry_distance * 0.25 +
            pp.successful_dribbles_leading_to_shot * 0.2 +
            pp.number_attempts_take_on_defender * 0.15 +
            (pp.aerial_duels_won + pp.aerial_duels_lost) * 0.15 
        ) AS high_intensity_score
    FROM players p
    JOIN player_performance pp ON p.player_id = pp.player_id
),
workload_percentiles AS (
    SELECT
        *,
        -- Calculate composite workload score
        (avg_minutes_per_game * 0.4 +
          dap90 * 0.4 + 
          oap90 * 0.3 +
          cap90 * 0.25
        ) AS composite_workload_score
    FROM player_workload
),
workload_percentiles_with_ranks AS (
    SELECT
        *,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY avg_minutes_per_game) AS minutes_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY oap90) AS oap90_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY dap90) AS dap90_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY cap90) AS cap90_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY high_intensity_score) AS high_intensity_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY discipline_score) AS discipline_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY progressive_actions_per_90) AS progressive_actions_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY fatigue_score) AS fatigue_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY composite_workload_score) AS composite_workload_score_percentile,
        
        -- Injury risk index
        (fatigue_score * 0.4 +
          discipline_score * 0.2 +
          age * 0.1 
        ) AS injury_risk_index
    FROM player_workload
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
    composite_workload_score,
    composite_workload_score_percentile,
    CASE
        WHEN minutes_percentile > 0.9 AND (dap90_percentile > 0.9 OR oap90_percentile > 0.9 OR cap90_percentile > 0.9) THEN 'VERY HIGH'
        WHEN minutes_percentile > 0.7 AND (dap90_percentile > 0.8 OR oap90_percentile > 0.8 OR cap90_percentile > 0.8) THEN 'HIGH'
        WHEN minutes_percentile > 0.5 THEN 'MODERATE'
        ELSE 'LOW'
    END AS workload_risk_level
FROM workload_percentiles_with_ranks
ORDER BY composite_workload_score DESC;


-- 2. Injury Risk Prediction Score
-- Combines multiple factors to calculate injury risk
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
            WHEN p.age <= 23 THEN 0.32
            WHEN p.age <= 27 THEN 0.29
            WHEN p.age <= 30 THEN 0.32
            WHEN p.age <= 33 THEN 0.35
            ELSE 0.38
        END AS age_risk_factor,
        
        -- Workload risk factor
        pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0) AS avg_minutes,

        -- Physical stress indicators
        pp.fouls_committed + pp.fouls_drawn AS total_fouls_involved,
        pp.tackles + pp.number_of_times_dribbled_past_by_opponent AS tackle_exposure,
        pp.aerial_duels_won + pp.aerial_duels_lost AS aerial_exposure,

        -- Fatigue 
        pp.errors_leading_to_opponent_shot AS fatigue_errors,
        pp.dispossessed_carries + pp.missed_carries AS ball_control_errors,

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
),
risk_percentiles AS (
    SELECT 
        *,
        PERCENT_RANK() OVER (ORDER BY avg_minutes) AS workload_percentile,
        PERCENT_RANK() OVER (ORDER BY total_fouls_involved) AS foul_involvement_percentile,
        PERCENT_RANK() OVER (ORDER BY tackle_exposure) AS tackle_risk_percentile,
        PERCENT_RANK() OVER (ORDER BY recovery_demand) AS recovery_demand_percentile,
        PERCENT_RANK() OVER (ORDER BY ball_control_errors) AS error_rate_percentile
    FROM player_risk_factors
),
injury_risk_scores AS (
    SELECT 
        player_id,
        player_name,
        age,
        position,
        team,
        season,
        
        -- Calculate weighted injury risk score
        (age_risk_factor * 0.20 +
            workload_percentile * 0.25 +
            foul_involvement_percentile * 0.15 +
            tackle_risk_percentile * 0.20 +
            recovery_demand_percentile * 0.10 +
            error_rate_percentile * 0.10
        ) AS injury_risk_score,
        
        -- Individual risk components
        age_risk_factor AS age_risk,
        workload_percentile AS workload_risk,
        foul_involvement_percentile AS contact_risk,
        tackle_risk_percentile AS tackle_risk,
        recovery_demand_percentile AS recovery_risk
        
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
ORDER BY injury_risk_score DESC;


-- 3. Position-Specific Performance Benchmarks
-- Identifies players performing outside normal ranges for their position
WITH position_benchmarks AS (
    SELECT 
        position,
        league,
        AVG(minutes_played) AS avg_position_minutes,
        STDDEV(minutes_played) AS stddev_position_minutes,
        AVG(tackles + interceptions) AS avg_defensive_actions,
        STDDEV(tackles + interceptions) AS stddev_defensive_actions,
        AVG(passes_attempted) AS avg_passes,
        STDDEV(passes_attempted) AS stddev_passes,
        AVG(carries) AS avg_carries,
        STDDEV(carries) AS stddev_carries,
        AVG(fouls_committed) AS avg_fouls,
        STDDEV(fouls_committed) AS stddev_fouls,
        AVG(aerial_duels_won + aerial_duels_lost) AS avg_aerial_duels,
        STDDEV(aerial_duels_won + aerial_duels_lost) AS stddev_aerial_duels
    FROM players p
    JOIN player_performance pp ON p.player_id = pp.player_id
    GROUP BY position, league
),
player_deviations AS (
    SELECT 
        p.player_id,
        p.player_name,
        p.position,
        p.team,
        p.league,
        pp.minutes_played,
        pp.tackles + pp.interceptions AS defensive_actions,
        pp.passes_attempted,
        pp.carries,
        pp.fouls_committed,
        
        -- Calculate z-scores (standard deviations from mean)
        (pp.minutes_played - pb.avg_position_minutes) / NULLIF(pb.stddev_position_minutes, 0) AS minutes_z_score,
        (pp.tackles + pp.interceptions - pb.avg_defensive_actions) / NULLIF(pb.stddev_defensive_actions, 0) AS defensive_z_score,
        (pp.passes_attempted - pb.avg_passes) / NULLIF(pb.stddev_passes, 0) AS passing_z_score,
        (pp.carries - pb.avg_carries) / NULLIF(pb.stddev_carries, 0) AS carrying_z_score,
        (pp.fouls_committed - pb.avg_fouls) / NULLIF(pb.stddev_fouls, 0) AS fouls_z_score,
        (pp.aerial_duels_won + pp.aerial_duels_lost - pb.avg_aerial_duels) / NULLIF(pb.stddev_aerial_duels, 0) AS aerial_z_score
        
    FROM players p
    JOIN player_performance pp ON p.player_id = pp.player_id
    JOIN position_benchmarks pb ON p.position = pb.position AND p.league = pb.league
)
SELECT 
    player_id,
    player_name,
    position,
    team,
    league,
    minutes_played,

    -- Flag outliers (> 2 standard deviations)
    CASE WHEN ABS(minutes_z_score) > 2 THEN 'OUTLIER' ELSE 'NORMAL' END AS minutes_status,
    CASE WHEN ABS(defensive_z_score) > 2 THEN 'OUTLIER' ELSE 'NORMAL' END AS defensive_status,
    CASE WHEN ABS(passing_z_score) > 2 THEN 'OUTLIER' ELSE 'NORMAL' END AS passing_status,
    CASE WHEN ABS(carrying_z_score) > 2 THEN 'OUTLIER' ELSE 'NORMAL' END AS carrying_status,
    CASE WHEN fouls_z_score > 2 THEN 'HIGH_RISK' ELSE 'NORMAL' END AS discipline_status,
    
    -- Overall outlier score
    ABS(minutes_z_score) + ABS(defensive_z_score) + ABS(passing_z_score) + 
    ABS(carrying_z_score) + ABS(fouls_z_score) AS total_deviation_score
    
FROM player_deviations
WHERE ABS(minutes_z_score) > 1.5
   OR ABS(defensive_z_score) > 1.5
   OR ABS(passing_z_score) > 1.5
   OR ABS(carrying_z_score) > 1.5
   OR fouls_z_score > 1.5
ORDER BY total_deviation_score DESC;


-- 4. Team Workload Distribution Analysis
-- Identifies teams with unbalanced workload distribution
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
ORDER BY coefficient_of_variation DESC;


-- 5. Recovery Time Requirements Analysis
-- Estimates recovery needs based on match intensity
WITH match_intensity_metrics AS (
    SELECT 
        p.player_id,
        p.player_name,
        p.position,
        p.age,
        pp.minutes_played,
        pp.games_played,
        
        -- High intensity action score
        (pp.tackles * 3 + 
         pp.aerial_duels_won * 2 + 
         pp.aerial_duels_lost * 2 +
         pp.fouls_committed * 2.5 +
         pp.progressive_carries * 1.5 +
         pp.number_attempts_take_on_defender * 1.5) AS high_intensity_score,
         
        -- Contact exposure score
        (pp.fouls_drawn * 2 + 
         pp.fouls_committed * 2 +
         pp.tackles_won + 
         pp.number_of_times_dribbled_past_by_opponent +
         pp.aerial_duels_won + 
         pp.aerial_duels_lost) AS contact_score,
         
        -- Sprint load estimate (based on progressive actions)
        (pp.progressive_carries * 3 + 
         pp.progressive_passes_count * 1 +
         pp.carries_into_18 * 2) AS sprint_load_score
         
    FROM players p
    JOIN player_performance pp ON p.player_id = pp.player_id
),
recovery_calculations AS (
    SELECT 
        player_id,
        player_name,
        position,
        age,
        minutes_played,
        games_played,
        
        -- Base recovery hours needed
        CASE 
            WHEN age <= 23 THEN 48
            WHEN age <= 27 THEN 56
            WHEN age <= 30 THEN 64
            ELSE 72
        END AS base_recovery_hours,
        
        -- Additional recovery based on intensity
        high_intensity_score / NULLIF(games_played, 0) * 0.5 AS intensity_recovery_hours,
        contact_score / NULLIF(games_played, 0) * 0.3 AS contact_recovery_hours,
        sprint_load_score / NULLIF(games_played, 0) * 0.2 AS sprint_recovery_hours,
        
        -- Minutes per game
        minutes_played * 1.0 / NULLIF(games_played, 0) AS avg_minutes_per_game
        
    FROM match_intensity_metrics
)
SELECT 
    player_id,
    player_name,
    position,
    age,
    avg_minutes_per_game,
    base_recovery_hours,
    intensity_recovery_hours,
    contact_recovery_hours,
    sprint_recovery_hours,
    
    -- Total recommended recovery time
    base_recovery_hours + 
    intensity_recovery_hours + 
    contact_recovery_hours + 
    sprint_recovery_hours AS total_recovery_hours,
    
    -- Recovery risk flag
    CASE 
        WHEN (base_recovery_hours + intensity_recovery_hours + contact_recovery_hours + sprint_recovery_hours) > 96 THEN 'EXTENDED_RECOVERY_NEEDED'
        WHEN (base_recovery_hours + intensity_recovery_hours + contact_recovery_hours + sprint_recovery_hours) > 72 THEN 'HIGH_RECOVERY_NEEDS'
        ELSE 'NORMAL_RECOVERY'
    END AS recovery_category
    
FROM recovery_calculations
WHERE avg_minutes_per_game > 45  -- Focus on regular players
ORDER BY total_recovery_hours DESC;


-- 6. Performance Efficiency Score
-- Identifies players with high output relative to physical load
WITH efficiency_metrics AS (
    SELECT 
        p.player_id,
        p.player_name,
        p.position,
        p.team,
        
        -- Output metrics
        pp.shot_creating_actions + pp.goal_creating_actions AS creative_actions,
        pp.tackles_won + pp.interceptions AS defensive_wins,
        pp.progressive_carries + pp.progressive_passes_count AS progressive_plays,
        
        -- Load metrics  
        pp.minutes_played AS total_minutes,
        pp.fouls_committed AS fouls,
        pp.yellow_cards + pp.red_cards * 2 AS cards_weighted,
        
        -- Calculate per-90 metrics
        CASE WHEN pp.minutes_played > 0 
            THEN (
              pp.assists + 
              pp.goals_per_shot_on_target +
              pp.shots_leading_to_another_shot +
              pp.successful_dribbles_leading_to_goal +
              pp.shots_leading_to_goal_scoring_shot
              ) * 90.0 / pp.minutes_played
            ELSE 0 
        END AS offensive_actions_per90,
        
        CASE WHEN pp.minutes_played > 0
            THEN (
              pp.assists + 
              pp.assisted_shots + 
              pp.progressive_passes_count + 
              pp.shot_creating_actions +
              pp.progressive_carries +
              pp.passes_attempted +
              pp.successful_dribbles_leading_to_shot
              ) * 90.0 / pp.minutes_played
            ELSE 0
        END AS creative_actions_per90,
        
        CASE WHEN pp.minutes_played > 0
            THEN (
              pp.tackles_won + 
              pp.interceptions +
              pp.clearances +
              pp.number_of_dribblers_tackled +
              pp.shots_blocked +
              pp.passes_blocked +
              pp.aerial_duels_won +
              pp.number_of_loose_balls_recovered
              ) * 90.0 / pp.minutes_played
            ELSE 0
        END AS defensive_actions_per90,
        
        CASE WHEN pp.minutes_played > 0
            THEN pp.fouls_committed * 90.0 / pp.minutes_played
            ELSE 0
        END AS fouls_per90
        
    FROM players p
    JOIN player_performance pp ON p.player_id = pp.player_id
    WHERE pp.minutes_played >= 400  -- Minimum playtime of 400 minutes
    AND pp.games_played >= 5        -- Minimum 5 game appearances
),
position_adjusted_efficiency AS (
    SELECT 
        e.*,
        
        -- Position-specific efficiency score
        CASE 
            -- Pure position players
            WHEN position = 'FW' THEN 
                (offensive_actions_per90 * 0.4 + creative_actions_per90 * 0.25 + defensive_actions_per90 * 0.15 - fouls_per90 * 0.2)
            WHEN position = 'MF' THEN
                (offensive_actions_per90 * 0.25 + creative_actions_per90 * 0.35 + defensive_actions_per90 * 0.2 - fouls_per90 * 0.2)
            WHEN position = 'DF' THEN
                (defensive_actions_per90 * 0.4 + creative_actions_per90 * 0.25 + offensive_actions_per90 * 0.15 - fouls_per90 * 0.2)
            
            -- Hybrid position players
            WHEN position = 'MFFW' OR position = 'FWMF' THEN
                -- Balanced between forward and midfielder weights
                (offensive_actions_per90 * 0.33 + creative_actions_per90 * 0.32 + defensive_actions_per90 * 0.15 - fouls_per90 * 0.2)
            WHEN position = 'MFDF' OR position = 'DFMF' THEN
                -- Balanced between midfielder and defender weights
                (defensive_actions_per90 * 0.32 + creative_actions_per90 * 0.33 + offensive_actions_per90 * 0.15 - fouls_per90 * 0.2)
            WHEN position = 'FWDF' OR position = 'DFFW' THEN
                -- Balanced between forward and defender weights
                (offensive_actions_per90 * 0.3 + defensive_actions_per90 * 0.3 + creative_actions_per90 * 0.2 - fouls_per90 * 0.2)

            WHEN position = 'GK' THEN 0  -- Goalkeepers need different metrics

            -- Default case for any other position combinations
            ELSE (offensive_actions_per90 * 0.3 + creative_actions_per90 * 0.3 + defensive_actions_per90 * 0.2 - fouls_per90 * 0.2)
        END AS efficiency_score,
        
        -- Rank within position
        RANK() OVER (PARTITION BY position ORDER BY 
            CASE 
                -- Pure positions with balanced weights
                WHEN position = 'FW' THEN 
                    (offensive_actions_per90 * 0.5 + creative_actions_per90 * 0.3 + defensive_actions_per90 * 0.2)
                WHEN position = 'MF' THEN 
                    (offensive_actions_per90 * 0.3 + creative_actions_per90 * 0.4 + defensive_actions_per90 * 0.3)
                WHEN position = 'DF' THEN 
                    (offensive_actions_per90 * 0.2 + creative_actions_per90 * 0.3 + defensive_actions_per90 * 0.5)
                    
                -- Hybrid positions with balanced weights
                WHEN position = 'MFFW' OR position = 'FWMF' THEN
                    (offensive_actions_per90 * 0.4 + creative_actions_per90 * 0.35 + defensive_actions_per90 * 0.25)
                WHEN position = 'MFDF' OR position = 'DFMF' THEN
                    (offensive_actions_per90 * 0.25 + creative_actions_per90 * 0.35 + defensive_actions_per90 * 0.4)
                WHEN position = 'FWDF' OR position = 'DFFW' THEN
                    (offensive_actions_per90 * 0.35 + creative_actions_per90 * 0.3 + defensive_actions_per90 * 0.35)
                    
                -- Default case with balanced weights
                ELSE (offensive_actions_per90 * 0.33 + creative_actions_per90 * 0.34 + defensive_actions_per90 * 0.33)
            END DESC
        ) AS position_rank
        
    FROM efficiency_metrics e
)
SELECT 
    player_id,
    player_name,
    position,
    team,
    total_minutes,
    offensive_actions_per90 AS offense_per90,
    creative_actions_per90 AS creativity_per90,
    defensive_actions_per90 AS defense_per90,
    fouls_per90 AS fouls_per90,
    efficiency_score AS efficiency_score,
    position_rank,
    
    CASE 
        WHEN position_rank <= 10 THEN 'ELITE'
        WHEN position_rank <= 20 THEN 'EXCELLENT'
        WHEN position_rank <= 30 THEN 'GOOD'
        ELSE 'AVERAGE'
    END AS performance_tier
    
FROM position_adjusted_efficiency
WHERE position != 'GK'  -- Exclude goalkeepers
ORDER BY efficiency_score DESC
LIMIT 50;


-- 7. League-Specific Performance Analysis
-- Analyzes player performance within their league context
WITH overall_player_rankings AS (
    -- Original query remains unchanged
    SELECT 
        player_id,
        player_name,
        position,
        team,
        league,  -- Make sure league is included in the original query
        total_minutes,
        offensive_actions_per90 AS offense_per90,
        creative_actions_per90 AS creativity_per90,
        defensive_actions_per90 AS defense_per90,
        fouls_per90 AS fouls_per90,
        efficiency_score AS efficiency_score,
        position_rank,
        CASE 
            WHEN position_rank <= 10 THEN 'ELITE'
            WHEN position_rank <= 20 THEN 'EXCELLENT'
            WHEN position_rank <= 30 THEN 'GOOD'
            ELSE 'AVERAGE'
        END AS performance_tier
    FROM position_adjusted_efficiency
    WHERE position != 'GK'
),
league_analysis AS (
    SELECT 
        league,
        position,
        player_name,
        team,
        efficiency_score,
        performance_tier,
        -- Rank within league and position
        RANK() OVER (
            PARTITION BY league, position 
            ORDER BY efficiency_score DESC
        ) AS league_position_rank,
        -- Count of players in this position in this league
        COUNT(*) OVER (
            PARTITION BY league, position
        ) AS players_in_position
    FROM overall_player_rankings
)
SELECT 
    league,
    position,
    player_name,
    team,
    efficiency_score,
    performance_tier,
    league_position_rank,
    players_in_position,
    -- Calculate percentile within league and position
    (league_position_rank::float / players_in_position) * 100 AS position_percentile,
    -- League-specific performance tier
    CASE 
        WHEN league_position_rank <= 5 THEN 'LEAGUE_ELITE'
        WHEN league_position_rank <= 10 THEN 'LEAGUE_EXCELLENT'
        WHEN league_position_rank <= 15 THEN 'LEAGUE_GOOD'
        ELSE 'LEAGUE_AVERAGE'
    END AS league_performance_tier
FROM league_analysis
ORDER BY 
    league,
    position,
    efficiency_score DESC;


-- 8. Injury Risk Alert System
-- Real-time monitoring query for high-risk players
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
            WHEN p.age <= 23 THEN 0.32
            WHEN p.age <= 27 THEN 0.29
            WHEN p.age <= 30 THEN 0.32
            WHEN p.age <= 33 THEN 0.35
            ELSE 0.38
        END AS age_risk_factor,
        
        -- Workload metrics
        pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0) AS avg_minutes,
        
        -- Physical stress indicators
        pp.fouls_committed + pp.fouls_drawn AS total_fouls_involved,
        pp.tackles + pp.number_of_times_dribbled_past_by_opponent AS tackle_exposure,
        pp.aerial_duels_won + pp.aerial_duels_lost AS aerial_exposure,
        
        -- Fatigue indicators
        pp.errors_leading_to_opponent_shot AS fatigue_errors,
        pp.dispossessed_carries + pp.missed_carries AS ball_control_errors,
        
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
    WHERE pp.minutes_played >= 400  -- Minimum 400 minutes played
    AND pp.games_played >= 5        -- Minimum 5 game appearances
),
risk_percentiles AS (
    SELECT 
        *,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY avg_minutes) AS workload_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY total_fouls_involved) AS foul_involvement_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY tackle_exposure) AS tackle_risk_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY recovery_demand) AS recovery_demand_percentile,
        PERCENT_RANK() OVER (PARTITION BY position ORDER BY ball_control_errors) AS error_rate_percentile
    FROM player_risk_factors
),
injury_risk_scores AS (
    SELECT 
        player_id,
        player_name,
        age,
        position,
        team,
        
        -- Calculate weighted injury risk score
        (age_risk_factor * 0.20 +
            workload_percentile * 0.25 +
            foul_involvement_percentile * 0.15 +
            tackle_risk_percentile * 0.20 +
            recovery_demand_percentile * 0.10 +
            error_rate_percentile * 0.10
        ) AS risk_score,
        
        -- Individual risk components
        age_risk_factor AS age_risk,
        workload_percentile AS workload_risk,
        foul_involvement_percentile AS contact_risk,
        tackle_risk_percentile AS tackle_risk,
        recovery_demand_percentile AS recovery_risk,
        error_rate_percentile AS fatigue_risk
        
    FROM risk_percentiles
)
SELECT 
    *,
    -- Risk level classification
    CASE 
        WHEN risk_score >= 0.8 THEN 'CRITICAL'   -- Top 20% risk
        WHEN risk_score >= 0.65 THEN 'VERY HIGH' -- Next 15% risk
        WHEN risk_score >= 0.5 THEN 'HIGH'       -- Next 15% risk
        WHEN risk_score >= 0.35 THEN 'MODERATE'  -- Next 15% risk
        ELSE 'LOW'                               -- Bottom 35% risk
    END AS risk_level,
    
    -- Specific recommendations based on risk factors
    CASE
        WHEN age_risk > 0.35 AND workload_risk > 0.8 THEN 'Reduce minutes - age/workload combination'
        WHEN fatigue_risk > 0.8 AND contact_risk > 0.8 THEN 'Full rest recommended - multiple risk factors'
        WHEN fatigue_risk > 0.8 THEN 'Light training only - showing fatigue signs'
        WHEN contact_risk > 0.8 THEN 'Limit contact in training'
        WHEN recovery_risk > 0.8 THEN 'Reduce physical load in training'
        ELSE 'Normal training'
    END AS recommendation,
    
    -- Risk factors present
    ARRAY_REMOVE(ARRAY[
        CASE WHEN age_risk > 0.35 THEN 'Age Risk' END,
        CASE WHEN workload_risk > 0.8 THEN 'High Workload' END,
        CASE WHEN fatigue_risk > 0.8 THEN 'Fatigue Signs' END,
        CASE WHEN contact_risk > 0.8 THEN 'High Contact' END,
        CASE WHEN recovery_risk > 0.8 THEN 'Recovery Needed' END
    ], NULL) AS active_risk_factors
    
FROM injury_risk_scores
WHERE risk_score >= 0.5  -- Only show high risk and above
ORDER BY risk_score DESC;