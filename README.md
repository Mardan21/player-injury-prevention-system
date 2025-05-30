# Player Performance & Injury Prevention System

## 🎯 Project Overview
An ML-powered system to predict football player injuries and optimize squad rotation using real data from Europe's top 5 leagues.

## 🛠️ Tech Stack
- **SQL**: PostgreSQL with complex window functions and CTEs
- **Python**: scikit-learn, pandas, SHAP for explainable AI
- **Visualization**: Tableau Public dashboards
- **Deployment**: Streamlit Cloud

## 📊 Current Progress
- [x] Project setup and structure
- [x] Data collection module
- [ ] Database schema
- [ ] SQL analytics queries
- [ ] ML prediction model
- [ ] Tableau dashboards
- [ ] Web application

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/player-injury-prevention-system.git
cd player-injury-prevention-system

# Set up environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run data collection
python python/data_collector.py