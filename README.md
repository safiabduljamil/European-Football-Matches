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
| **Home advantage trends** | Murat Arikan | - Calculate home win % by league<br>- Analyze historical trends<br>- Statistical testing |
| **League excitement rankings** | Abdul Jamil Safi | - Count 0-0 draws<br>- Calculate avg. goals/match<br>- Rank leagues by excitement |
| **Seasonal goal patterns** | Leandro Da Silva Pinto | - Monthly goal averages<br>- Yearly comparisons<br>- Weather impact analysis |
| *Bonus: Match prediction* | All Members | - ML model development<br>- Feature engineering<br>- Accuracy validation |

## 🛠️ Technical Implementation

### 📦 Dependencies
``bash
#Core Requirements
dash==2.14.1
pandas==2.1.4
plotly==5.18.0
# Abgabetermin: 30.03.2025

Veränderer: Murat - 28.03.2025

Was getan wurde:

• In den Zeilen 4, 5, 6, 7, 8 und 9 habe ich eine Datenbereinigung, Informationsabfrage und Überprüfung durchgeführt.
                 
• In den Zeilen 10 und 11 habe ich die Anzahl der Siege überprüft, um den Heimvorteil zu analysieren, und die Ergebnisse mit einem Kreisdiagramm visualisiert.
               
• In den Zeilen 12 und 13 habe ich die Spalte mit dem Jahr, die als String vorlag, in ein Datumsformat umgewandelt und auf die Jahreszahlen zugegriffen.
              
• In den Zeilen 14 und 15 habe ich die Veränderung der Heimsiege im Jahresverlauf berechnet und visualisiert.

• In Zeile 16 habe ich eine visuelle Linie hinzugefügt.

• In Zeile 17 habe ich den durchschnittlichen Siegwert und die Varianz berechnet.

• In Zeile 18 habe ich die Abnahmerate der Siege und die Steigung des Trends berechnet.

• In den Zeilen 19, 20 und 21 wurde dem DataFrame eine neue Spalte mit dem Namen „Land“ hinzugefügt. In Zeile 21 kann man sie sehen.

• In den Zeilen 22, 23 und 24 habe ich berechnet und visualisiert, in welchem Land der Heimvorteil am stärksten ausgeprägt ist.

• In den Zeilen 24 und 25 habe ich berechnet und visualisiert, wie sich der Heimvorteil je nach Land verändert hat. (Die Grafik ist allerdings ziemlich unübersichtlich geworden.)

• In den Zeilen 27 und 28 habe ich, weil die vorherige Grafik unübersichtlich war, eine neue Visualisierung erstellt, die die Trend-Steigungen pro Land zeigt. Sie ist verständlicher geworden (glaube ich jedenfalls :) ).
