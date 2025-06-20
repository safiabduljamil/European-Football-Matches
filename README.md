# ⚽ European Football Matches Analysis Dashboard

![Dashboard Screenshot](screenshot.png) *Add actual dashboard screenshot*

## 📋 Project Overview
Interactive web dashboard analyzing European football statistics with visualizations of:
- Home advantage trends across leagues
- 0-0 draw frequencies and league excitement
- Seasonal goal patterns
- Head-to-head team comparisons

## 👥 Team Responsibilities

### 🔍 Research Questions

| Question | Assignee | Key Tasks |
|----------|----------|-----------|
| **League excitement rankings** | Abdul Jamil Safi | - Count 0-0 draws<br>- Calculate avg. goals/match<br>- Rank leagues by excitement |
| **Home advantage trends** | Murat Arikan | - Calculate home win % by league<br>- Analyze historical trends<br>- Statistical testing |
| **Seasonal goal patterns** | Leandro Da Silva Pinto | - Monthly goal averages<br>- Yearly comparisons<br>- Weather impact analysis |
| *Bonus: Match prediction* | All Members | - ML model development<br>- Feature engineering<br>- Accuracy validation |

## 🛠️ Technical Implementation

### 📦 Dependencies

#Core Requirements
dash==2.14.1
pandas==2.1.4
plotly==5.18.0

# ⚡ Dash App Overview

This project is built using [Dash](https://dash.plotly.com/), a powerful Python framework for building interactive web applications and dashboards — entirely in Python. Dash is ideal for data visualization, data analysis, and ML apps, especially for users familiar with Python and Plotly.
---
## 📦 Requirements

To run this Dash app, make sure the following Python packages are installed:
dash
dash-bootstrap-components
pandas
plotly
scikit-learn

## 📊 Dataset Description

The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/flynn28/european-football-matches) and includes historical data from 22 European football leagues.

### Dataset Columns:
- `League`: Name of the football league
- `Date`: Date of the match
- `HomeTeam`: Name of the home team
- `AwayTeam`: Name of the away team
- `HomeGoals`: Number of goals by home team
- `AwayGoals`: Number of goals by away team
- `Result`: Match result (H = Home win, A = Away win, D = Draw)

**Data Size:**  
- 23 CSV tables (22 individual leagues + 1 merged file)  
- Approx. 22.0 MB in total  
- All files follow the same 7-column structure

## 🔧 Installation Instructions

1. **Clone the repository** (or download it as a ZIP):
git clone https://github.com/safiabduljamil/European-Football-Matches.git
Navigate to the dashboard folder where the requirements.txt file is located:

cd  dashboard
Install the required Python packages:

pip install -r requirements.txt

💡 Note: Make sure you have Python 3.10 or higher installed before running the above command.

## 👥 Project Contributors

- **Abdul Jamil Safi** – Projektleiter  
  📧 abdul.safi@stud.fhgr.ch

- **Murat Arikan** – Statistical analysis  
  📧 murat.arikan@stud.fhgr.ch

- **Leandro Da Silva Pinto** – Visual analytics  
  📧 leandro.dasilvapinto@stud.fhgr.ch
```bash

