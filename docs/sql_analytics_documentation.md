# SQL Analytics Documentation
## Player Workload & Injury Risk Analysis System

### Overview
This system provides comprehensive analytics for player workload management and injury risk prediction in professional soccer. It consists of 8 specialized SQL queries and a Python testing framework designed to identify at-risk players, optimize training loads, and prevent injuries.

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│   SQL Analytics  │───▶│   Risk Reports  │
│   (Players &    │    │   (8 Queries)    │    │   & Alerts      │
│   Performance)  │    └──────────────────┘    └─────────────────┘
└─────────────────┘              │
                                 ▼
                    ┌──────────────────┐
                    │  Materialized    │
                    │  Views & Indexes │
                    └──────────────────┘
```

---

## Database Schema Requirements

### Core Tables

#### `players` Table
| Column | Type | Description |
|--------|------|-------------|
| player_id | INTEGER | Primary key |
| player_name | VARCHAR | Player full name |
| age | INTEGER | Current age |
| position | VARCHAR | Playing position (FW, MF, DF, GK, etc.) |
| team | VARCHAR | Current team name |
| league | VARCHAR | League/competition name |

#### `player_performance` Table
| Column | Type | Description |
|--------|------|-------------|
| player_id | INTEGER | Foreign key to players |
| minutes_played | INTEGER | Total minutes played |
| games_played | INTEGER | Number of games appeared |
| goals | INTEGER | Goals scored |
| assists | INTEGER | Assists provided |
| tackles | INTEGER | Tackles attempted |
| tackles_won | INTEGER | Successful tackles |
| interceptions | INTEGER | Interceptions made |
| clearances | INTEGER | Clearances made |
| fouls_committed | INTEGER | Fouls committed |
| fouls_drawn | INTEGER | Fouls drawn |
| yellow_cards | INTEGER | Yellow cards received |
| red_cards | INTEGER | Red cards received |
| progressive_carries | INTEGER | Progressive ball carries |
| progressive_passes_count | INTEGER | Progressive passes |
| aerial_duels_won | INTEGER | Aerial duels won |
| aerial_duels_lost | INTEGER | Aerial duels lost |
| shots_on_target | INTEGER | Shots on target |
| carries | INTEGER | Total ball carries |
| passes_attempted | INTEGER | Total passes attempted |
| *[Additional performance metrics...]* | | |

---

## Analytics Queries

### 1. Player Workload Intensity Score
**Purpose:** Calculates comprehensive workload metrics and identifies players at risk of overexertion.

**Key Metrics:**
- **Defensive Actions Per 90 (DAP90):** Weighted score of defensive contributions
- **Creative Actions Per 90 (CAP90):** Creative and playmaking contributions  
- **Offensive Actions Per 90 (OAP90):** Goal-scoring and offensive contributions
- **Composite Workload Score:** Combined metric across all areas
- **Fatigue Score:** Physical demand assessment

**Risk Levels:**
- `VERY HIGH`: Minutes >90th percentile + Any action category >90th percentile
- `HIGH`: Minutes >70th percentile + Any action category >80th percentile  
- `MODERATE`: Minutes >50th percentile
- `LOW`: Below moderate thresholds

**Output:** Player rankings with workload percentiles and risk classifications

### 2. Injury Risk Prediction Score
**Purpose:** Multi-factor injury risk assessment combining age, workload, and physical stress indicators.

**Risk Factors:**
- **Age Risk (20% weight):** Age-adjusted risk curve
- **Workload Risk (25% weight):** Minutes per game percentile
- **Contact Risk (15% weight):** Fouls and tackles involvement
- **Recovery Demand (10% weight):** High-intensity action requirements
- **Error Rate (10% weight):** Fatigue-related mistakes

**Risk Categories:**
- `CRITICAL`: Top 20% risk (≥0.8 score)
- `VERY HIGH`: Next 15% risk (0.65-0.79)
- `HIGH`: Next 15% risk (0.5-0.64)
- `MODERATE`: Next 15% risk (0.35-0.49)
- `LOW`: Bottom 35% risk (<0.35)

### 3. Position-Specific Performance Benchmarks
**Purpose:** Identifies players performing outside normal statistical ranges for their position.

**Analysis:**
- Z-score calculations for key metrics by position
- Outlier detection (>2 standard deviations)
- Cross-league position comparisons

**Flags:**
- `OUTLIER`: Performance >2 standard deviations from position mean
- `HIGH_RISK`: Discipline issues above normal range
- `NORMAL`: Within expected performance range

### 4. Team Workload Distribution Analysis
**Purpose:** Identifies teams with unbalanced workload distribution that may indicate over-reliance on key players.

**Metrics:**
- **Coefficient of Variation:** Workload inequality measure
- **Minutes Range:** Difference between highest and lowest player minutes
- **Workload Balance Score:** Gini coefficient adaptation

**Risk Assessment:**
- `HIGH`: Coefficient of variation >50%
- `MODERATE`: Coefficient of variation 30-50%
- `LOW`: Coefficient of variation <30%

### 5. Recovery Time Requirements Analysis
**Purpose:** Estimates recovery needs based on match intensity and player characteristics.

**Factors:**
- **Base Recovery:** Age-adjusted baseline (48-72 hours)
- **Intensity Recovery:** High-intensity actions impact
- **Contact Recovery:** Physical contact exposure impact
- **Sprint Recovery:** Running load impact

**Categories:**
- `EXTENDED_RECOVERY_NEEDED`: >96 hours total recovery
- `HIGH_RECOVERY_NEEDS`: 72-96 hours total recovery  
- `NORMAL_RECOVERY`: <72 hours total recovery

### 6. Performance Efficiency Score
**Purpose:** Identifies players with high output relative to physical load.

**Position-Weighted Scoring:**
- **Forwards (FW):** Offensive 40%, Creative 25%, Defensive 15%
- **Midfielders (MF):** Creative 35%, Offensive 25%, Defensive 20%
- **Defenders (DF):** Defensive 40%, Creative 25%, Offensive 15%
- **Hybrid Positions:** Balanced weighting between primary roles

**Performance Tiers:**
- `ELITE`: Top 10 in position
- `EXCELLENT`: Top 11-20 in position
- `GOOD`: Top 21-30 in position
- `AVERAGE`: Below top 30

### 7. League-Specific Performance Analysis
**Purpose:** Analyzes player performance within their specific league context.

**Features:**
- League and position-specific percentile rankings
- Cross-league performance comparisons
- League-adjusted performance tiers

### 8. Injury Risk Alert System
**Purpose:** Real-time monitoring for immediate risk identification.

**Alert Triggers:**
- Risk score ≥0.5 (High risk and above)
- Multiple risk factors present simultaneously
- Fatigue indicators above 80th percentile

**Recommendations:**
- `Reduce minutes`: Age/workload combination risk
- `Full rest recommended`: Multiple high-risk factors
- `Light training only`: Fatigue signs detected
- `Limit contact in training`: High contact exposure
- `Normal training`: Low risk profile

---

## Python Testing Framework

### Core Components

#### `QueryTester` Class
Main testing orchestrator with methods for:
- Individual query testing
- Result validation and export
- Materialized view creation
- Summary report generation

#### Key Methods

##### `test_workload_query()`
- Tests workload intensity calculations
- Exports top 50 players by composite workload score
- Validates risk level classifications

##### `test_injury_risk_query()`
- Tests injury prediction algorithms
- Exports top 50 highest risk players
- Validates risk category assignments

##### `test_team_analysis_query()`
- Tests team workload distribution
- Identifies teams with highest workload imbalance
- Exports coefficient of variation rankings

##### `create_materialized_view()`
- Creates `player_risk_summary` materialized view
- Adds performance indexes
- Enables fast risk lookups

##### `generate_summary_report()`
- Consolidates all analysis results
- Generates executive summary with key metrics
- Provides actionable recommendations

---

## Usage Instructions

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure database connection
cp .env.template .env
# Edit .env with your database credentials

# Test connection
python db_config.py
```

### Running Analytics
```bash
# Run full analysis suite
python test_queries.py

# Individual query testing
python -c "from test_queries import QueryTester; QueryTester().test_workload_query()"
```

### Output Files
All results are saved to the `outputs/` directory:
- `test_analysis.csv`: Workload analysis results
- `injury_risk_analysis.csv`: Injury risk predictions
- `team_workload_distribution.csv`: Team analysis results
- `analysis_summary.json`: Executive summary report

---

## Key Performance Indicators

### Player Level
- **Workload Risk Level:** Distribution across risk categories
- **Injury Risk Score:** 0-1 scale with category classifications
- **Performance Efficiency:** Position-adjusted output metrics
- **Recovery Requirements:** Hours needed between matches

### Team Level  
- **Workload Distribution:** Coefficient of variation
- **High-Risk Player Count:** Players in critical/very high risk categories
- **Squad Depth Utilization:** Distribution of minutes across squad
- **Position-Specific Risks:** Risk concentration by playing position

### League Level
- **Performance Benchmarks:** Position-specific statistical ranges
- **Risk Distribution:** League-wide injury risk patterns
- **Efficiency Leaders:** Top performers by efficiency metrics

---

## Maintenance & Updates

### Materialized View Refresh
```sql
REFRESH MATERIALIZED VIEW player_risk_summary;
```

### Index Optimization
```sql
-- Weekly index maintenance
REINDEX INDEX idx_risk_summary_risk;
ANALYZE player_risk_summary;
```

### Data Quality Checks
- Validate player performance data completeness
- Check for statistical outliers requiring investigation
- Monitor query performance and optimization needs

---

## Integration Points

### External Systems
- **Training Load Monitoring:** Export risk scores for training planning
- **Medical Systems:** Integrate risk alerts with injury tracking
- **Performance Analysis:** Provide context for match analysis
- **Squad Management:** Support rotation and selection decisions

### API Endpoints (Future)
- `GET /players/{id}/risk`: Individual player risk assessment
- `GET /teams/{id}/workload`: Team workload distribution
- `GET /alerts/high-risk`: Current high-risk player alerts
- `GET /reports/weekly`: Weekly analysis summary

---

## Troubleshooting

### Common Issues
1. **Missing Data:** Ensure all required performance metrics are populated
2. **Query Performance:** Check indexes and consider data partitioning
3. **Risk Score Anomalies:** Validate input data quality and scoring weights
4. **Position Classifications:** Verify position mapping consistency

### Performance Optimization
- Use materialized views for frequently accessed data
- Implement proper indexing strategy
- Consider data archiving for historical seasons
- Monitor query execution plans

---

## Version History
- **v1.0:** Initial implementation with 8 core analytics queries
- **v1.1:** Added materialized views and Python testing framework
- **v1.2:** Enhanced risk scoring algorithms and position-specific analysis