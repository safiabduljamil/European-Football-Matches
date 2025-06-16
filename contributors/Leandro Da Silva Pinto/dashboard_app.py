import os
import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

#  Load & preprocess data
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "merged_data.csv")
df = pd.read_csv(csv_path)

# Parse dates (DD/MM/YY), drop bad rows, compute total goals
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%y", errors='coerce')
df = df.dropna(subset=['Date', 'HomeGoals', 'AwayGoals'])
df['TotalGoals'] = df['HomeGoals'] + df['AwayGoals']
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Define season string (e.g. "2022/2023")
def get_season(row):
    y, m = row['Year'], row['Month']
    return f"{y}/{y+1}" if m >= 7 else f"{y-1}/{y}"
df['Season'] = df.apply(get_season, axis=1)

# Compute summaries
monthly_avg = df.groupby('Month')['TotalGoals'].mean().reset_index()
monthly_avg['MonthName'] = pd.to_datetime(monthly_avg['Month'], format='%m') \
                               .dt.strftime('%b')
season_avg = df.groupby('Season')['TotalGoals'].mean().reset_index()

# Build Dash app 
app = dash.Dash(__name__, title="Tor-Trends Dashboard")

def serve_layout():
    return html.Div([
        html.H1("Analyse der Tor-Trends", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.H2("Durchschnittliche Tore pro Monat"),
                dcc.Graph(
                    figure=px.bar(
                        monthly_avg,
                        x='MonthName', y='TotalGoals',
                        labels={'MonthName': 'Monat', 'TotalGoals': 'Tore pro Spiel'},
                        color='TotalGoals',
                        color_continuous_scale='viridis',
                        title="Monatliche Durchschnittswerte"
                    ).update_layout(
                        xaxis=dict(title='Monat'),
                        yaxis=dict(title='Tore pro Spiel'),
                        coloraxis_showscale=False
                    )
                )
            ], className='six columns'),

            html.Div([
                html.H2("Durchschnittliche Tore pro Saison"),
                dcc.Graph(
                    figure=px.line(
                        season_avg,
                        x='Season', y='TotalGoals',
                        labels={'Season': 'Saison', 'TotalGoals': 'Tore pro Spiel'},
                        markers=True,
                        title="Saisonale Durchschnittswerte"
                    ).update_layout(
                        xaxis=dict(title='Saison', tickangle=45),
                        yaxis=dict(title='Tore pro Spiel')
                    )
                )
            ], className='six columns'),

        ], className='row', style={'padding': '20px'})
    ])

app.layout = serve_layout()

# Run 
if __name__ == '__main__':
    app.run(debug=True)