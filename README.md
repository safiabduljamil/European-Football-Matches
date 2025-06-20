# ⚽ European Football Matches Analysis Dashboard

## 📋 Project Overview
Interactive web dashboard analyzing European football statistics with visualizations of:
- Home advantage trends across leagues
- 0-0 draw frequencies and league excitement
- Seasonal goal patterns
- Head-to-head team comparisons

### 🔍👥 Research Questions and Project Contributors

- **Abdul Jamil Safi** – Project Lead  
  🔹 Focus: Research Question 2  
  🔹 Tasks: 0:0 analysis, SpannungsScore, league comparison  
  📧 abdul.safi@stud.fhgr.ch

- **Murat Arikan**  
  🔹 Focus: Research Question 1  
  🔹 Tasks: Home advantage trends, statistical testing  
  📧 murat.arikan@stud.fhgr.ch

- **Leandro Da Silva Pinto**  
  🔹 Focus: Research Question 3  
  🔹 Tasks: Seasonal goal trends, month-by-month analysis  
  📧 leandro.dasilvapinto@stud.fhgr.ch

- **All Members**  
  🔹 Focus: Research Question 4 (Bonus)  
  🔹 Tasks: Match prediction, model integration into dashboard


## 🛠️ Technical Implementation

### 📦 Dependencies

#Core Requirements
dash==2.14.1
pandas==2.1.4
plotly==5.18.0

# ⚡ Dash App Overview

This project is built using [Dash](https://dash.plotly.com/), a powerful Python framework for building interactive web applications and dashboards entirely in Python. Dash is ideal for data visualization, data analysis, and ML apps, especially for users familiar with Python and Plotly.
---
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
## 📦 Requirements

To run this Dash app, make sure the following Python packages are installed:
dash
dash-bootstrap-components
pandas
plotly
scikit-learn 

The requirements.txt file is in the Dashboard folder.
pip install -r requirements.txt 

## 🚀 Running the Dashboard
python app.py

##screenshots
![127 0 0 1_8050_](https://github.com/user-attachments/assets/89b2a9e5-888e-4b6b-90b2-c2afc4aa388c)
![127 0 0 1_8050_ (1)](https://github.com/user-attachments/assets/ac13222c-a493-451b-b18a-ffb47d568ed8)
![127 0 0 1_8050_ (2)](https://github.com/user-attachments/assets/f1205a00-4cd1-45d1-aed3-312b0619e889)
![127 0 0 1_8050_ (3)](https://github.com/user-attachments/assets/4be6b63c-b10c-4ae0-9a57-8535936a17ac)


💡 Note: Ensure you have Python 3.10 or higher installed before running the above command.
```bash

