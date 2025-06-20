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
``bash
#Core Requirements
dash==2.14.1
pandas==2.1.4
plotly==5.18.0

## 📝 Changelog - Home Advantage Analysis (Murat Arikan)  
**Last Updated:** March 28, 2025  
### 🔧 Implemented Improvements  

#### Data Processing  
✅ **Lines 4-9**:  
- Performed data cleaning and validation  
- Executed information queries and consistency checks  

#### Visualization  
✅ **Lines 10-11**:  
- Analyzed home advantage through win counts  
- Created pie chart visualizations of results  

#### Date Conversion  
✅ **Lines 12-13**:  
- Converted year column from string to datetime format  
- Extracted year values for temporal analysis  

#### Trend Analysis  
✅ **Lines 14-18**:  
- Calculated yearly home win trends  
- Added trendline visualization  
- Computed:  
  - Average win values  
  - Variance metrics  
  - Win rate decline slope  

#### Country-Level Analysis  
✅ **Lines 19-21**:  
- Added new "Country" column to DataFrame  

✅ **Lines 22-24**:  
- Identified countries with the strongest home advantage  
- Visualized cross-country comparisons  

#### Enhanced Visualizations  
⚠️ **Lines 24-25**:  
- Initial country trend visualization proved cluttered  

✅ **Lines 27-28**:  
- Implemented improved visualization showing:  
  - Per-country trend slopes  
  - Clear comparative analysis  
