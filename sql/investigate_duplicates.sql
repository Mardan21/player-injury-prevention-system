-- Fix duplicate performance records by aggregating them
-- Use this if the duplicates are truly duplicate data that should be combined

-- Create a temporary table with aggregated performance data
CREATE TEMP TABLE aggregated_performance AS
SELECT 
    player_id,
    
    -- Sum up the counting stats (goals, assists, etc.)
    SUM(minutes_played) as minutes_played,
    SUM(games_played) as games_played,
    SUM(games_started) as games_started,
    SUM(goals) as goals,
    SUM(assists) as assists,
    SUM(shots) as shots,
    SUM(shots_on_target) as shots_on_target,
    SUM(passes_completed) as passes_completed,
    SUM(passes_attempted) as passes_attempted,
    SUM(total_passes_distance) as total_passes_distance,
    SUM(touches) as touches,
    SUM(tackles) as tackles,
    SUM(interceptions) as interceptions,
    SUM(yellow_cards) as yellow_cards,
    SUM(red_cards) as red_cards,
    SUM(fouls_committed) as fouls_committed,
    SUM(fouls_drawn) as fouls_drawn,
    
    -- Calculate averages for percentage stats
    CASE 
        WHEN SUM(passes_attempted) > 0 
        THEN ROUND(SUM(passes_completed) * 100.0 / SUM(passes_attempted), 2)
        ELSE 0 
    END as pass_completion_percentage,
    
    -- Keep the most recent created_at
    MAX(created_at) as created_at
    
FROM player_performance
GROUP BY player_id;

-- Show what we're about to do
SELECT 
    'BEFORE aggregation:' as status,
    COUNT(*) as performance_records,
    COUNT(DISTINCT player_id) as unique_players
FROM player_performance;

SELECT 
    'AFTER aggregation:' as status,
    COUNT(*) as performance_records,
    COUNT(DISTINCT player_id) as unique_players
FROM aggregated_performance;

-- Replace the original performance data with aggregated data
-- (Uncomment these lines after reviewing the above results)

-- /*
-- -- Delete original performance data
-- DELETE FROM player_performance;

-- -- Insert aggregated data
-- INSERT INTO player_performance (
--     player_id, minutes_played, games_played, games_started,
--     goals, assists, shots, shots_on_target,
--     passes_completed, passes_attempted, pass_completion_percentage,
--     total_passes_distance, touches, tackles, interceptions,
--     yellow_cards, red_cards, fouls_committed, fouls_drawn, created_at
-- )
-- SELECT 
--     player_id, minutes_played, games_played, games_started,
--     goals, assists, shots, shots_on_target,
--     passes_completed, passes_attempted, pass_completion_percentage,
--     total_passes_distance, touches, tackles, interceptions,
--     yellow_cards, red_cards, fouls_committed, fouls_drawn, created_at
-- FROM aggregated_performance;

-- -- Verify the fix
-- SELECT 
--     'FINAL RESULT:' as status,
--     (SELECT COUNT(*) FROM players) as players,
--     (SELECT COUNT(*) FROM player_performance) as performance_records;
-- */