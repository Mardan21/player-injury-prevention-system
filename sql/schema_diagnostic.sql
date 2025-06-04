-- Schema Diagnostic Script
-- Run this to check your current database schema

-- Check if tables exist and their structure
SELECT 
    table_name,
    column_name,
    data_type,
    character_maximum_length,
    numeric_precision,
    numeric_scale
FROM information_schema.columns 
WHERE table_name IN ('players', 'player_performance', 'injury_events', 'player_injury_summary')
ORDER BY table_name, ordinal_position;

-- Check for any DECIMAL(4,2) fields that might be causing overflow
SELECT 
    table_name,
    column_name,
    data_type,
    numeric_precision,
    numeric_scale
FROM information_schema.columns 
WHERE numeric_precision = 4 AND numeric_scale = 2
ORDER BY table_name, column_name;

-- Check current data in player_performance to see what's failing
SELECT 
    column_name,
    MAX(LENGTH(column_name::text)) as max_length
FROM player_performance 
LIMIT 1;

-- Show any constraints that might be causing issues
SELECT
    tc.constraint_name,
    tc.table_name,
    kcu.column_name,
    tc.constraint_type
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu 
    ON tc.constraint_name = kcu.constraint_name
WHERE tc.table_name IN ('players', 'player_performance')
ORDER BY tc.table_name, kcu.column_name;