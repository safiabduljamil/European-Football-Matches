# European-Football-Matches
# Project Kickoff

Projekttitel: Analyse europäischer Fussballspiele
Forschungsfragen & Aufgabenzuweisung
Um eine effiziente Bearbeitung des Projekts zu gewährleisten, haben wir die Forschungsfragen unter den Teammitgliedern aufgeteilt. Jedes Mitglied ist für eine spezifische Frage verantwortlich, basierend auf seinen Stärken. Falls jemand zusätzlich an einer anderen Frage arbeiten möchte, ist das ebenfalls in Ordnung. Eine zusätzliche optionale Forschungsfrage steht allen Teammitgliedern zur Verfügung.
Wir werden die Arbeit gemeinsam in einem GitHub-Repository organisieren, sodass jeder Änderungen hochladen und einsehen kann. Bitte stellt sicher, dass ihr regelmäßig eure Fortschritte im Repository aktualisiert.

1) Gibt es einen Heimvorteil in allen europäischen Ligen, und hat sich dieser im Laufe der Jahre verringert?

Zugewiesen an: Murat Arikan 

Anweisungen:
•	Berechne den Prozentsatz der Heimsiege pro Liga über die Jahre.
•	Identifiziere Trends: Hat sich der Heimvorteil verändert?
•	Führe eine statistische Analyse durch (Mittelwert, Varianz, Trendanalyse).
•	Präsentiere die Ergebnisse in Tabellen oder Grafiken zur besseren Visualisierung.
•  und ...


2) Welche Liga hat die wenigsten 0:0-Spiele, und welche Liga ist die spannendste?
   
Zugewiesen an: Abdul Jamil Safi 

Anweisungen:
•	Schreibe Python-Skripte, um 0:0-Spiele pro Liga zu filtern und zu zählen.
•	Berechne die durchschnittliche Anzahl an Toren pro Spiel für jede Liga.
•	Erstelle ein Ranking der spannendsten Ligen (Ligen mit mehr Toren sind spannender).
•	Visualisiere die Ergebnisse mit matplotlib/seaborn.
•  und ...

3) In welchem Monat oder welcher Saison werden die meisten Tore erzielt, und beeinflusst die Saison die Anzahl der Tore?
   
Zugewiesen an: Leandro Da Silva Pinto 

Anweisungen:
•	Extrahiere relevante Daten und analysiere Tor-Trends pro Monat/Saison.
•	Interpretiere und erkläre, wie sich die Jahreszeiten auf die Toranzahl auswirken.
•	Schreibe eine verständliche Erklärung der Ergebnisse für den Abschlussbericht.
•	Stelle sicher, dass die Ergebnisse für Leser leicht verständlich sind.
•  und ...

4) Inwieweit können die Ergebnisse zukünftiger Fußballspiele durch die Analyse
historischer Spieldaten mit statistischen Methoden prognostiziert werden?  
Zusätzliche optionale Forschungsfrage
   
(Jedes Teammitglied kann daran arbeiten)

•  Bereinige und bereite historische Spieldaten vor (Heimteam, Auswärtsteam, Tore, etc.).
•  Erstelle zusätzliche Merkmale wie Teamform, Ligaposition, Head-to-Head-Ergebnisse.
•  Verwende statistische Modelle (z. B. logistische Regression, Random Forest) oder maschinelles Lernen (z. B. SVM, Gradient Boosting) zur Prognose von Spielergebnissen.

•  Teile die Daten in Trainings- und Testdaten auf, trainiere das Modell und validiere es.
•  Prognostiziere die Ergebnisse zukünftiger Spiele und bewerte die Modellgenauigkeit (z. B. Accuracy, F1-Score).
•  Visualisiere die Vorhersagen und überprüfe die Modellleistung.

Allgemeine Richtlinien:
•	Jedes Teammitglied sollte seinen Prozess und seine Ergebnisse dokumentieren.
•	Visualisierungen und statistische Ergebnisse sollten gut beschriftet sein.
•	Der Abschlussbericht sollte eine Zusammenfassung aller Ergebnisse enthalten.
•	Zusammenarbeit wird empfohlen: Besprecht Herausforderungen und Erkenntnisse im Team.
•	Alle Fortschritte und Ergebnisse sollten regelmäßig im GitHub-Repository hochgeladen werden.
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
