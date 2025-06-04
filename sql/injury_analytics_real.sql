-- injury_analytics_real.sql
-- Advanced analytics using real injury data from Transfermarkt via worldfootballR

-- 1. Current Injury Risk Assessment with Historical Validation
WITH player_risk_analysis AS (
    SELECT 
        p.player_id,
        p.player_name,
        p.age,
        p.position,
        p.team,
        
        -- Current workload
        pp.minutes_played,
        pp.games_played,
        pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0) AS avg_minutes,
        
        -- Actual injury history
        COALESCE(pis.total_injuries, 0) AS career_injuries,
        COALESCE(pis.injury_prone_score, 0) AS injury_prone_score,
        COALESCE(pis.injury_frequency, 0) AS injuries_per_year,
        pis.last_injury_date,
        
        -- Days since last injury
        CASE 
            WHEN pis.last_injury_date IS NOT NULL 
            THEN DATE_PART('day', CURRENT_DATE - pis.last_injury_date)
            ELSE 999
        END AS days_since_injury,
        
        -- Recent injury count
        (SELECT COUNT(*) FROM injury_events ie 
         WHERE ie.player_id = p.player_id 
         AND ie.injury_date >= CURRENT_DATE - INTERVAL '180 days') as injuries_last_6_months,
        
        -- Physical load
        pp.tackles + pp.aerial_duels_won + pp.aerial_duels_lost AS physical_actions,
        
        -- Calculate evidence-based risk score
        ROUND(
            -- Historical injury risk (40% weight - most predictive)
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
        ) AS risk_score
        
    FROM players p
    JOIN player_performance pp ON p.player_id = pp.player_id
    LEFT JOIN player_injury_summary pis ON p.player_id = pis.player_id
)
SELECT 
    player_id,
    player_name,
    age,
    position,
    team,
    ROUND(avg_minutes, 1) AS avg_minutes_per_game,
    career_injuries,
    injuries_per_year,
    days_since_injury,
    injuries_last_6_months AS recent_injuries,
    risk_score,
    CASE 
        WHEN risk_score >= 0.7 THEN 'CRITICAL'
        WHEN risk_score >= 0.5 THEN 'HIGH'
        WHEN risk_score >= 0.3 THEN 'MODERATE'
        ELSE 'LOW'
    END AS risk_level,
    CASE 
        WHEN injuries_last_6_months > 0 AND days_since_injury < 60 THEN 'Active injury concerns - maximum caution'
        WHEN career_injuries > 5 THEN 'Injury prone - enhanced prevention protocol'
        WHEN days_since_injury < 90 THEN 'Recent return - graduated loading'
        WHEN injuries_per_year > 2 THEN 'High frequency - rotation recommended'
        ELSE 'Standard monitoring'
    END AS recommendation
FROM player_risk_analysis
ORDER BY risk_score DESC
LIMIT 50;

-- 2. Injury Pattern Analysis by Type and Position
WITH injury_patterns AS (
    SELECT 
        p.position,
        ie.injury_category,
        COUNT(*) as injury_count,
        AVG(ie.days_out) as avg_recovery_days,
        MIN(ie.days_out) as min_recovery,
        MAX(ie.days_out) as max_recovery,
        STDDEV(ie.days_out) as recovery_stddev
    FROM injury_events ie
    JOIN players p ON ie.player_id = p.player_id
    WHERE ie.season IN ('2023/24', '2022/23')
    GROUP BY p.position, ie.injury_category
    HAVING COUNT(*) >= 5  -- Minimum sample size
),
position_totals AS (
    SELECT 
        position,
        COUNT(*) as total_position_injuries
    FROM injury_events ie
    JOIN players p ON ie.player_id = p.player_id
    WHERE ie.season IN ('2023/24', '2022/23')
    GROUP BY position
)
SELECT 
    ip.position,
    ip.injury_category,
    ip.injury_count,
    ROUND(ip.injury_count * 100.0 / pt.total_position_injuries, 1) as pct_of_position_injuries,
    ROUND(ip.avg_recovery_days, 1) as avg_recovery_days,
    ip.min_recovery || '-' || ip.max_recovery as recovery_range,
    ROUND(ip.recovery_stddev, 1) as recovery_variability,
    CASE 
        WHEN ip.avg_recovery_days > 30 THEN 'Severe'
        WHEN ip.avg_recovery_days > 14 THEN 'Moderate'
        ELSE 'Minor'
    END as typical_severity
FROM injury_patterns ip
JOIN position_totals pt ON ip.position = pt.position
ORDER BY ip.position, ip.injury_count DESC;

-- 3. Team Injury Analysis with Actual Data
WITH team_injury_stats AS (
    SELECT 
        p.team,
        COUNT(DISTINCT p.player_id) as squad_size,
        COUNT(DISTINCT CASE WHEN pis.total_injuries > 0 THEN p.player_id END) as players_with_injuries,
        SUM(COALESCE(pis.total_injuries, 0)) as total_injuries,
        SUM(COALESCE(pis.total_days_injured, 0)) as total_days_lost,
        AVG(COALESCE(pis.injury_prone_score, 0)) as avg_injury_prone_score,
        
        -- Current injuries
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM injury_events ie 
                WHERE ie.player_id = p.player_id 
                AND ie.return_date > CURRENT_DATE
            ) THEN p.player_id 
        END) as currently_injured,
        
        -- Key players injured (high minutes)
        COUNT(DISTINCT CASE 
            WHEN pis.total_injuries > 0 
            AND pp.minutes_played > (SELECT AVG(minutes_played) FROM player_performance)
            THEN p.player_id 
        END) as key_players_injured
        
    FROM players p
    LEFT JOIN player_injury_summary pis ON p.player_id = pis.player_id
    LEFT JOIN player_performance pp ON p.player_id = pp.player_id
    GROUP BY p.team
)
SELECT 
    team,
    squad_size,
    players_with_injuries,
    ROUND(players_with_injuries * 100.0 / squad_size, 1) || '%' as injury_rate,
    total_injuries,
    total_days_lost,
    ROUND(total_days_lost * 1.0 / NULLIF(total_injuries, 0), 1) as avg_days_per_injury,
    currently_injured,
    key_players_injured,
    ROUND(avg_injury_prone_score, 3) as team_injury_risk,
    RANK() OVER (ORDER BY total_days_lost DESC) as injury_burden_rank,
    CASE 
        WHEN avg_injury_prone_score > 0.5 THEN 'High injury risk squad'
        WHEN avg_injury_prone_score > 0.3 THEN 'Moderate injury risk'
        ELSE 'Low injury risk'
    END as team_risk_assessment
FROM team_injury_stats
ORDER BY total_days_lost DESC;

-- 4. Muscle Injury Analysis (Most Common Type)
WITH muscle_injuries AS (
    SELECT 
        p.player_id,
        p.player_name,
        p.age,
        p.position,
        COUNT(*) as muscle_injury_count,
        SUM(ie.days_out) as total_days_out,
        AVG(ie.days_out) as avg_recovery,
        STRING_AGG(ie.injury_type, '; ' ORDER BY ie.injury_date) as injury_details,
        MIN(ie.injury_date) as first_injury,
        MAX(ie.injury_date) as last_injury
    FROM injury_events ie
    JOIN players p ON ie.player_id = p.player_id
    WHERE ie.injury_category IN ('muscle', 'hamstring', 'calf', 'groin')
    AND ie.season IN ('2023/24', '2022/23')
    GROUP BY p.player_id, p.player_name, p.age, p.position
    HAVING COUNT(*) >= 2  -- Recurring muscle injuries
)
SELECT 
    player_name,
    age,
    position,
    muscle_injury_count,
    total_days_out,
    ROUND(avg_recovery, 1) as avg_recovery_days,
    DATE_PART('day', last_injury - first_injury) as days_between_first_last,
    ROUND(
        DATE_PART('day', last_injury - first_injury) * 1.0 / NULLIF(muscle_injury_count - 1, 0), 
        1
    ) as avg_days_between_injuries,
    CASE 
        WHEN muscle_injury_count >= 3 THEN 'Chronic muscle injury problem'
        WHEN avg_recovery > 30 THEN 'Severe muscle injuries'
        ELSE 'Recurring muscle strain'
    END as injury_pattern,
    injury_details
FROM muscle_injuries
ORDER BY muscle_injury_count DESC, total_days_out DESC
LIMIT 20;

-- 5. Injury Recovery Time Analysis
WITH recovery_analysis AS (
    SELECT 
        ie.injury_category,
        ie.severity,
        COUNT(*) as injury_count,
        AVG(ie.days_out) as avg_days,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ie.days_out) as q1_days,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ie.days_out) as median_days,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ie.days_out) as q3_days,
        MIN(ie.days_out) as min_days,
        MAX(ie.days_out) as max_days
    FROM injury_events ie
    WHERE ie.days_out IS NOT NULL
    AND ie.days_out > 0
    GROUP BY ie.injury_category, ie.severity
    HAVING COUNT(*) >= 10
)
SELECT 
    injury_category,
    severity,
    injury_count,
    ROUND(avg_days, 1) as avg_recovery_days,
    ROUND(median_days, 1) as median_recovery_days,
    ROUND(q1_days, 1) || ' - ' || ROUND(q3_days, 1) as interquartile_range,
    min_days || ' - ' || max_days as full_range,
    CASE 
        WHEN median_days <= 7 THEN 'Quick recovery expected'
        WHEN median_days <= 21 THEN 'Standard recovery period'
        WHEN median_days <= 42 THEN 'Extended recovery needed'
        ELSE 'Long-term rehabilitation'
    END as recovery_expectation
FROM recovery_analysis
ORDER BY injury_category, severity;

-- 6. Young Player Injury Trends
WITH young_player_injuries AS (
    SELECT 
        p.player_id,
        p.player_name,
        p.age,
        p.position,
        p.team,
        pis.total_injuries,
        pis.total_days_injured,
        pis.injury_prone_score,
        pp.minutes_played,
        pp.games_played
    FROM players p
    JOIN player_injury_summary pis ON p.player_id = pis.player_id
    JOIN player_performance pp ON p.player_id = pp.player_id
    WHERE p.age <= 23
    AND pis.total_injuries > 0
),
age_group_comparison AS (
    SELECT 
        CASE 
            WHEN age <= 21 THEN 'U21'
            ELSE 'U23'
        END as age_group,
        COUNT(DISTINCT player_id) as player_count,
        AVG(total_injuries) as avg_injuries,
        AVG(total_days_injured) as avg_days_injured,
        AVG(injury_prone_score) as avg_injury_risk,
        AVG(minutes_played * 1.0 / NULLIF(games_played, 0)) as avg_minutes_per_game
    FROM young_player_injuries
    GROUP BY age_group
)
SELECT 
    ypi.player_name,
    ypi.age,
    ypi.position,
    ypi.team,
    ypi.total_injuries,
    ypi.total_days_injured,
    ROUND(ypi.injury_prone_score, 3) as injury_risk,
    ROUND(ypi.minutes_played * 1.0 / NULLIF(ypi.games_played, 0), 1) as avg_minutes,
    CASE 
        WHEN ypi.injury_prone_score > 0.5 THEN 'High risk for young player'
        WHEN ypi.total_injuries > 2 THEN 'Concerning injury frequency'
        WHEN ypi.total_days_injured > 60 THEN 'Significant time lost'
        ELSE 'Monitor development'
    END as assessment
FROM young_player_injuries ypi
ORDER BY ypi.injury_prone_score DESC
LIMIT 20;

-- 7. Create Real-Time Monitoring View
CREATE OR REPLACE VIEW injury_risk_monitoring_real AS
SELECT 
    p.player_id,
    p.player_name,
    p.age,
    p.position,
    p.team,
    
    -- Current workload
    pp.minutes_played,
    pp.games_played,
    ROUND(pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0), 1) as avg_minutes,
    
    -- Injury history
    COALESCE(pis.total_injuries, 0) as career_injuries,
    COALESCE(pis.injury_prone_score, 0) as injury_prone_score,
    COALESCE(pis.injury_frequency, 0) as injuries_per_year,
    
    -- Recent injuries
    (SELECT COUNT(*) FROM injury_events ie 
     WHERE ie.player_id = p.player_id 
     AND ie.injury_date >= CURRENT_DATE - INTERVAL '90 days') as recent_injuries,
    
    -- Days since last injury
    CASE 
        WHEN pis.last_injury_date IS NOT NULL 
        THEN DATE_PART('day', CURRENT_DATE - pis.last_injury_date)
        ELSE 999
    END as days_since_injury,
    
    -- Currently injured?
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM injury_events ie 
            WHERE ie.player_id = p.player_id 
            AND ie.return_date > CURRENT_DATE
        ) THEN TRUE ELSE FALSE
    END as currently_injured,
    
    -- Risk calculation
    ROUND(
        COALESCE(pis.injury_prone_score, 0) * 0.4 +
        CASE WHEN pp.minutes_played * 1.0 / NULLIF(pp.games_played, 0) > 80 THEN 0.2 ELSE 0.1 END +
        CASE WHEN p.age > 30 THEN 0.2 WHEN p.age > 28 THEN 0.1 ELSE 0.05 END +
        CASE 
            WHEN pis.last_injury_date > CURRENT_DATE - INTERVAL '90 days' THEN 0.2
            WHEN pis.last_injury_date > CURRENT_DATE - INTERVAL '180 days' THEN 0.1
            ELSE 0
        END,
        3
    ) as current_risk_score,
    
    CURRENT_TIMESTAMP as last_updated
    
FROM players p
JOIN player_performance pp ON p.player_id = pp.player_id
LEFT JOIN player_injury_summary pis ON p.player_id = pis.player_id;