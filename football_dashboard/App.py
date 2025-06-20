import os
import glob
import numpy as np
import pandas as pd
import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ------------------------
# Data Loading and Preparation
# ------------------------

def load_match_data():
    """Load all CSV files from the shared-data folder"""
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "../shared-data/*.csv")
    all_files = glob.glob(data_path)
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # Data cleaning and processing
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date', 'HomeGoals', 'AwayGoals'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['TotalGoals'] = df['HomeGoals'] + df['AwayGoals']
    
    # Add country information
    league_to_country = {
        'Bundesliga': 'Germany',
        'Bundesliga 2': 'Germany',
        'English Premier League': 'England',
        'English Championship': 'England',
        'English League 1': 'England',
        'English League 2': 'England',
        'English Conference': 'England',
        'La Liga': 'Spain',
        'La Liga 2': 'Spain',
        'Serie A': 'Italy',
        'Serie B': 'Italy',
        'Ligue 1': 'France',
        'Ligue 2': 'France',
        'Scottish Premier League': 'Scotland',
        'Scottish Division 1': 'Scotland',
        'Scottish Division 2': 'Scotland',
        'Scottish Division 3': 'Scotland',
        'Greek Super League': 'Greece',
        'Jupiler League': 'Belgium',
        'Liga Portugal': 'Portugal',
        'Eredivisie': 'Netherlands',
        'Super Lig': 'Turkey'
    }
    df['Country'] = df['League'].map(league_to_country)
    
    # Define season string (e.g. "2022/2023")
    df['Season'] = df.apply(lambda row: f"{row['Year']}/{row['Year']+1}" 
                           if row['Month'] >= 7 else f"{row['Year']-1}/{row['Year']}", axis=1)
    
    return df

def compute_league_stats(df):
    """Calculate league statistics including 0-0 draws and SpannungScore"""
    goalless_df = df[(df['HomeGoals'] == 0) & (df['AwayGoals'] == 0)]
    league_matches = df.groupby('League').size().reset_index(name='TotalMatches')
    league_goalless = goalless_df.groupby('League').size().reset_index(name='GoallessMatches')
    league_stats = pd.merge(league_matches, league_goalless, on='League', how='left').fillna(0)
    league_stats['GoallessPercentage'] = (league_stats['GoallessMatches'] / league_stats['TotalMatches']) * 100

    league_stats['AvgGoals'] = league_stats.apply(
        lambda row: (df[df['League'] == row['League']]['HomeGoals'].sum()
                     + df[df['League'] == row['League']]['AwayGoals'].sum()) / row['TotalMatches'], axis=1)

    league_stats['CloseGamesPercent'] = league_stats.apply(
        lambda row: 100 * ((df[df['League'] == row['League']]
                            .assign(goal_diff=lambda x: abs(x['HomeGoals'] - x['AwayGoals']))['goal_diff'] == 1).sum()
                          / row['TotalMatches']), axis=1)

    league_stats['SpannungsScore'] = league_stats['AvgGoals'] + (league_stats['CloseGamesPercent'] / 10)
    league_stats = league_stats.round({'AvgGoals': 2, 'CloseGamesPercent': 2, 'SpannungsScore': 2, 'GoallessPercentage': 2})
    return league_stats.sort_values('SpannungsScore', ascending=False)

def get_head_to_head(df, team1, team2):
    """Get historical match results between two teams"""
    mask = ((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) | \
           ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))
    matches = df[mask]
    if matches.empty:
        return None, matches
    
    team1_wins = ((matches['HomeTeam'] == team1) & (matches['HomeGoals'] > matches['AwayGoals']) |
                  ((matches['AwayTeam'] == team1) & (matches['AwayGoals'] > matches['HomeGoals']))).sum()
    team2_wins = ((matches['HomeTeam'] == team2) & (matches['HomeGoals'] > matches['AwayGoals']) |
                  ((matches['AwayTeam'] == team2) & (matches['AwayGoals'] > matches['HomeGoals']))).sum()
    draws = (matches['HomeGoals'] == matches['AwayGoals']).sum()
    total = len(matches)
    
    return {
        'team1_wins': int(team1_wins),
        'team2_wins': int(team2_wins),
        'draws': int(draws),
        'total_matches': total
    }, matches

def prepare_home_advantage_data(df):
    """Prepare data for home advantage analysis"""
    # Overall stats
    total_matches = df.shape[0]
    home_wins = (df['HomeGoals'] > df['AwayGoals']).sum()
    away_wins = (df['HomeGoals'] < df['AwayGoals']).sum()
    draws = (df['HomeGoals'] == df['AwayGoals']).sum()
    
    home_win_pct = (home_wins / total_matches) * 100
    away_win_pct = (away_wins / total_matches) * 100
    draw_pct = (draws / total_matches) * 100
    
    avg_home_goals = df['HomeGoals'].mean()
    avg_away_goals = df['AwayGoals'].mean()
    
    # Yearly stats
    yearly_stats = df.groupby('Year').agg(
        total_matches=('HomeGoals', 'count'),
        home_wins=('HomeGoals', lambda x: (x > df.loc[x.index, 'AwayGoals']).sum()),
        away_wins=('AwayGoals', lambda x: (x > df.loc[x.index, 'HomeGoals']).sum()),
        avg_home_goals=('HomeGoals', 'mean'),
        avg_away_goals=('AwayGoals', 'mean')
    )
    
    yearly_stats['home_win_percentage'] = (yearly_stats['home_wins'] / yearly_stats['total_matches']) * 100
    yearly_stats['away_win_percentage'] = (yearly_stats['away_wins'] / yearly_stats['total_matches']) * 100
    
    # Country stats
    country_stats = df.groupby('Country').agg(
        total_matches=('HomeGoals', 'count'),
        home_wins=('HomeGoals', lambda x: (x > df.loc[x.index, 'AwayGoals']).sum()),
        away_wins=('AwayGoals', lambda x: (x > df.loc[x.index, 'HomeGoals']).sum()),
        avg_home_goals=('HomeGoals', 'mean'),
        avg_away_goals=('AwayGoals', 'mean')
    )
    
    country_stats['home_win_pct'] = (country_stats['home_wins'] / country_stats['total_matches']) * 100
    country_stats['away_win_pct'] = (country_stats['away_wins'] / country_stats['total_matches']) * 100
    country_stats['draw_pct'] = ((country_stats['total_matches'] - country_stats['home_wins'] - country_stats['away_wins']) / 
                                 country_stats['total_matches']) * 100
    
    country_stats_sorted = country_stats.sort_values(by='home_win_pct', ascending=False)
    
    return {
        'total_matches': total_matches,
        'home_wins': home_wins,
        'away_wins': away_wins,
        'draws': draws,
        'home_win_pct': home_win_pct,
        'away_win_pct': away_win_pct,
        'draw_pct': draw_pct,
        'avg_home_goals': avg_home_goals,
        'avg_away_goals': avg_away_goals,
        'yearly_stats': yearly_stats,
        'country_stats': country_stats_sorted
    }

def prepare_seasonal_data(df):
    """Prepare data for seasonal goal trends analysis"""
    monthly_avg = df.groupby('Month')['TotalGoals'].mean().reset_index()
    monthly_avg['MonthName'] = pd.to_datetime(monthly_avg['Month'], format='%m').dt.strftime('%b')
    season_avg = df.groupby('Season')['TotalGoals'].mean().reset_index()
    
    return {
        'monthly_avg': monthly_avg,
        'season_avg': season_avg
    }

# Load and prepare all data
matches_df = load_match_data()
league_stats = compute_league_stats(matches_df)
league_zero_draw_rate = league_stats.set_index('League')['GoallessPercentage'].to_dict()
league_spannung = league_stats.set_index('League')['SpannungsScore'].to_dict()
home_advantage_data = prepare_home_advantage_data(matches_df)
seasonal_data = prepare_seasonal_data(matches_df)

# ------------------------
# App Setup
# ------------------------

app = Dash(__name__, 
           external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/bWLwgP.css'], 
           suppress_callback_exceptions=True)

app.title = "Football Statistics Dashboard"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Football Statistics Dashboard", 
                    className="text-center mb-4"),
            width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Tabs(id="main-tabs", value='tab-league', children=[
                # Tab 1: 0:0 Draws & Excitement
                dcc.Tab(label="0:0 Draws & Excitement", value='tab-league', className='custom-tab', children=[
                    html.Br(),
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("League Analysis", className="card-title"),
                            html.P("Select leagues to compare 0:0 draw rates and excitement scores", 
                                  className="card-text"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select League(s):"),
                                    dcc.Dropdown(
                                        id='league-select',
                                        options=[{'label': lg, 'value': lg} 
                                                 for lg in sorted(league_stats['League'].unique())],
                                        multi=True,
                                        value=['English Premier League', 'Bundesliga', 'La Liga']
                                    )
                                ], width=12)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='zerozero-graph'), width=6),
                                dbc.Col(dcc.Graph(id='spannung-graph'), width=6)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col(html.Div(id='league-summary'), width=12)
                            ])
                        ])
                    ])
                ]),
                
                # Tab 2: Home Advantage Analysis
                dcc.Tab(label="Home Advantage Analysis", value='tab-home', className='custom-tab', children=[
                    html.Br(),
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Home Advantage Statistics", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Overall Statistics"),
                                    html.P(f"Total matches analyzed: {home_advantage_data['total_matches']:,}"),
                                    html.P(f"Home wins: {home_advantage_data['home_wins']:,} ({home_advantage_data['home_win_pct']:.2f}%)"),
                                    html.P(f"Away wins: {home_advantage_data['away_wins']:,} ({home_advantage_data['away_win_pct']:.2f}%)"),
                                    html.P(f"Draws: {home_advantage_data['draws']:,} ({home_advantage_data['draw_pct']:.2f}%)"),
                                    html.P(f"Average home goals: {home_advantage_data['avg_home_goals']:.2f}"),
                                    html.P(f"Average away goals: {home_advantage_data['avg_away_goals']:.2f}"),
                                ], width=4),
                                dbc.Col(dcc.Graph(id='win-rate-chart'), width=4),
                                dbc.Col(dcc.Graph(id='goals-chart'), width=4)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='win-rate-trend-chart'), width=6),
                                dbc.Col(dcc.Graph(id='goals-trend-chart'), width=6)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='country-comparison-chart'), width=12)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Key Findings"),
                                    html.Ul([
                                        html.Li("Home teams win significantly more often than away teams"),
                                        html.Li("Home teams score more goals on average than away teams"),
                                        html.Li("The home advantage has been decreasing over time"),
                                        html.Li("Greece shows the strongest home advantage, while Scotland shows the weakest"),
                                    ])
                                ], width=12)
                            ])
                        ])
                    ])
                ]),
                
                # Tab 3: Seasonal Goal Trends
                dcc.Tab(label="Seasonal Goal Trends", value='tab-season', className='custom-tab', children=[
                    html.Br(),
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Goal Trends by Month and Season", className="card-title"),
                            dbc.Row([
                                dbc.Col(dcc.Graph(
                                    figure=px.bar(
                                        seasonal_data['monthly_avg'],
                                        x='MonthName', y='TotalGoals',
                                        labels={'MonthName': 'Month', 'TotalGoals': 'Goals per Game'},
                                        color='TotalGoals',
                                        color_continuous_scale='viridis',
                                        title="Monthly Goal Averages"
                                    ).update_layout(
                                        xaxis=dict(title='Month'),
                                        yaxis=dict(title='Goals per Game'),
                                        coloraxis_showscale=False
                                    )
                                ), width=6),
                                dbc.Col(dcc.Graph(
                                    figure=px.line(
                                        seasonal_data['season_avg'],
                                        x='Season', y='TotalGoals',
                                        labels={'Season': 'Season', 'TotalGoals': 'Goals per Game'},
                                        markers=True,
                                        title="Seasonal Goal Averages"
                                    ).update_layout(
                                        xaxis=dict(title='Season', tickangle=45),
                                        yaxis=dict(title='Goals per Game')
                                    )
                                ), width=6)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Key Insights"),
                                    html.Ul([
                                        html.Li("Goals tend to be highest in the winter months (December-January)"),
                                        html.Li("Goal scoring has generally increased over the years"),
                                        html.Li("Some seasons show significant variation from the trend"),
                                    ])
                                ], width=12)
                            ])
                        ])
                    ])
                ]),
                
                # Tab 4: Head-to-Head Prediction
                dcc.Tab(label="Head-to-Head Prediction", value='tab-h2h', className='custom-tab', children=[
                    html.Br(),
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Team Comparison Tool", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select League:"),
                                    dcc.Dropdown(
                                        id='league-select-h2h',
                                        options=[{'label': lg, 'value': lg} 
                                                 for lg in sorted(league_stats['League'].unique())],
                                        value='English Premier League',
                                        placeholder="Choose a league"
                                    )
                                ], width=12)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Team 1:"),
                                    dcc.Dropdown(
                                        id='team1-select',
                                        options=[],
                                        value=None,
                                        placeholder="Team 1"
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Select Team 2:"),
                                    dcc.Dropdown(
                                        id='team2-select',
                                        options=[],
                                        value=None,
                                        placeholder="Team 2"
                                    )
                                ], width=6)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col(html.Div(id='h2h-summary'), width=12)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='h2h-graph'), width=6),
                                dbc.Col(dcc.Graph(id='h2h-line-graph'), width=6)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='h2h-home-away'), width=6),
                                dbc.Col(dcc.Graph(id='h2h-seasonal'), width=6)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col(html.Div(id='h2h-extra-impact'), width=12)
                            ])
                        ])
                    ])
                ])
            ])  # This closes the dcc.Tabs
        ], width=12)  # This closes the dbc.Col
    ])  # This closes the dbc.Row
], fluid=True)  # This closes the dbc.Container

# ------------------------
# Callbacks
# ------------------------

@app.callback(
    [Output('zerozero-graph', 'figure'),
     Output('spannung-graph', 'figure'),
     Output('league-summary', 'children')],
    Input('league-select', 'value')
)
def update_league_graphs(selected_leagues):
    if not selected_leagues:
        return px.bar(title="Select at least one league"), px.bar(title=""), ""
    
    filtered = league_stats[league_stats['League'].isin(selected_leagues)].sort_values('GoallessPercentage')
    
    fig_zerozero = px.bar(
        filtered,
        x='League',
        y='GoallessPercentage',
        title='0:0 Draw Percentage by League',
        labels={'GoallessPercentage': '0:0 Draw %'},
        color='GoallessPercentage',
        color_continuous_scale='Viridis'
    ).update_layout(yaxis_range=[0, 10])
    
    filtered = filtered.sort_values('SpannungsScore', ascending=False)
    fig_spannung = px.bar(
        filtered,
        x='League',
        y='SpannungsScore',
        title='Excitement Score by League',
        labels={'SpannungsScore': 'Excitement Score'},
        color='SpannungsScore',
        color_continuous_scale='Plasma'
    )
    
    best_league = filtered.iloc[0]['League']
    best_score = filtered.iloc[0]['SpannungsScore']
    worst_league = filtered.iloc[-1]['League']
    worst_score = filtered.iloc[-1]['SpannungsScore']
    
    summary = dbc.Alert([
        html.H5("Key Insights", className="alert-heading"),
        html.P(f"Most exciting league: {best_league} (Score: {best_score:.2f})"),
        html.P(f"Least exciting league: {worst_league} (Score: {worst_score:.2f})"),
        html.P("Excitement Score combines average goals and percentage of close games (1-goal difference)")
    ], color="info")
    
    return fig_zerozero, fig_spannung, summary

# Add these callbacks to your app
@app.callback(
    [Output('win-rate-chart', 'figure'),
     Output('goals-chart', 'figure'),
     Output('win-rate-trend-chart', 'figure'),
     Output('goals-trend-chart', 'figure'),
     Output('country-comparison-chart', 'figure')],
    Input('main-tabs', 'value')
)
def update_home_advantage_graphs(tab):
    if tab != 'tab-home':
        return [go.Figure()] * 5  # Return empty figures if not on home tab
    
    # Win rate bar chart
    win_rate_fig = go.Figure([
        go.Bar(
            x=['Home Wins', 'Away Wins', 'Draws'],
            y=[home_advantage_data['home_win_pct'], 
               home_advantage_data['away_win_pct'], 
               home_advantage_data['draw_pct']],
            marker_color=['royalblue', 'darkorange', 'gold'],
            text=[f"{home_advantage_data['home_win_pct']:.2f}%", 
                  f"{home_advantage_data['away_win_pct']:.2f}%", 
                  f"{home_advantage_data['draw_pct']:.2f}%"],
            textposition='auto'
        )
    ])
    win_rate_fig.update_layout(
        title='Win Rate: Home vs Away Teams',
        yaxis_title='Percentage (%)',
        yaxis=dict(range=[0, 100]),
        template='plotly_white'
    )
    
    # Goals bar chart
    goals_fig = go.Figure([
        go.Bar(
            x=['Home Team', 'Away Team'],
            y=[home_advantage_data['avg_home_goals'], 
               home_advantage_data['avg_away_goals']],
            marker_color=['royalblue', 'darkorange'],
            text=[f"{home_advantage_data['avg_home_goals']:.2f}", 
                  f"{home_advantage_data['avg_away_goals']:.2f}"],
            textposition='auto'
        )
    ])
    goals_fig.update_layout(
        title='Average Goals: Home vs Away Teams',
        yaxis_title='Average Goals Per Match',
        template='plotly_white'
    )
    
    # Win rate trend chart
    yearly_stats = home_advantage_data['yearly_stats']
    years = yearly_stats.index
    home = yearly_stats['home_win_percentage']
    away = yearly_stats['away_win_percentage']
    
    z_home = np.polyfit(years, home, 1)
    p_home = np.poly1d(z_home)
    z_away = np.polyfit(years, away, 1)
    p_away = np.poly1d(z_away)
    
    win_trend_fig = go.Figure()
    win_trend_fig.add_trace(go.Scatter(x=years, y=home, mode='lines+markers', name='Home Win %'))
    win_trend_fig.add_trace(go.Scatter(x=years, y=away, mode='lines+markers', name='Away Win %'))
    win_trend_fig.add_trace(go.Scatter(x=years, y=p_home(years), mode='lines', 
                                 line=dict(dash='dash', color='red'), name='Home Trend'))
    win_trend_fig.add_trace(go.Scatter(x=years, y=p_away(years), mode='lines', 
                                 line=dict(dash='dash', color='blue'), name='Away Trend'))
    win_trend_fig.update_layout(
        title='Win Rate: Home vs Away Teams Over the Years',
        xaxis_title='Year',
        yaxis_title='Win Percentage (%)',
        template='plotly_white'
    )
    
    # Goals trend chart
    home_goals = yearly_stats['avg_home_goals']
    away_goals = yearly_stats['avg_away_goals']
    
    z_home_goals = np.polyfit(years, home_goals, 1)
    p_home_goals = np.poly1d(z_home_goals)
    z_away_goals = np.polyfit(years, away_goals, 1)
    p_away_goals = np.poly1d(z_away_goals)
    
    goals_trend_fig = go.Figure()
    goals_trend_fig.add_trace(go.Scatter(x=years, y=home_goals, mode='lines+markers', name='Home Goals'))
    goals_trend_fig.add_trace(go.Scatter(x=years, y=away_goals, mode='lines+markers', name='Away Goals'))
    goals_trend_fig.add_trace(go.Scatter(x=years, y=p_home_goals(years), mode='lines', 
                                      name='Home Trend', line=dict(dash='dash', color='red')))
    goals_trend_fig.add_trace(go.Scatter(x=years, y=p_away_goals(years), mode='lines', 
                                      name='Away Trend', line=dict(dash='dash', color='blue')))
    goals_trend_fig.update_layout(
        title='Average Goals: Home vs Away Over the Years',
        xaxis_title='Year',
        yaxis_title='Average Goals',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99)
    )
    
    # Country comparison chart
    country_stats = home_advantage_data['country_stats']
    country_fig = go.Figure()
    country_fig.add_trace(go.Bar(
        x=country_stats.index,
        y=country_stats['home_win_pct'],
        name='Home Win %',
        marker_color='royalblue',
        text=country_stats['home_win_pct'].round(2),
        textposition='outside',
        texttemplate='%{text:.2f}%'

        
    ))
    country_fig.add_trace(go.Bar(
        x=country_stats.index,
        y=country_stats['away_win_pct'],
        name='Away Win %',
        marker_color='darkorange',
        text=country_stats['away_win_pct'].round(2),
        textposition='outside',
        texttemplate='%{text:.2f}%'
    ))
    country_fig.update_layout(
        barmode='group',
        title='Win Rate: Home vs Away by Country',
        xaxis_title='Country',
        yaxis_title='Percentage (%)',
        yaxis=dict(range=[0, 60]),
        template='plotly_white'
    )
    

    return win_rate_fig, goals_fig, win_trend_fig, goals_trend_fig, country_fig

@app.callback(
    [Output('team1-select', 'options'),
     Output('team2-select', 'options')],
    Input('league-select-h2h', 'value')
)
def update_team_dropdowns(selected_league):
    if not selected_league:
        return [], []
    teams = sorted(set(matches_df[matches_df['League'] == selected_league]['HomeTeam'].unique()) |
                   set(matches_df[matches_df['League'] == selected_league]['AwayTeam'].unique()))
    options = [{'label': team, 'value': team} for team in teams]
    return options, options

@app.callback(
    [Output('h2h-summary', 'children'),
     Output('h2h-graph', 'figure'),
     Output('h2h-line-graph', 'figure'),
     Output('h2h-extra-impact', 'children'),
     Output('h2h-home-away', 'figure'),
     Output('h2h-seasonal', 'figure')],
    [Input('team1-select', 'value'),
     Input('team2-select', 'value'),
     Input('league-select-h2h', 'value')],
    prevent_initial_call=True
)
def update_head_to_head(team1, team2, league):
    # Early return for empty states
    if not all([team1, team2, league]) or team1 == team2:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Select two different teams")
        return (
            dbc.Alert("Please select two different teams.", color="warning"),
            empty_fig,
            empty_fig,
            "",
            empty_fig,
            empty_fig
        )

    try:
        df = matches_df[matches_df['League'] == league]
        result, h2h_matches = get_head_to_head(df, team1, team2)
        
        if not result or h2h_matches.empty:
            raise ValueError("No historical data found")
            
        # Make sure we're working with a copy
        h2h_matches = h2h_matches.copy()

        def safe_calculate_team_spannung(team, league_df):
            try:
                team_matches = league_df[(league_df['HomeTeam'] == team) | (league_df['AwayTeam'] == team)]
                if len(team_matches) == 0:
                    return 0.0
                    
                total_goals = team_matches.apply(
                    lambda row: row['HomeGoals'] if row['HomeTeam'] == team else row['AwayGoals'], 
                    axis=1
                ).sum()
                
                avg_goals = total_goals / len(team_matches)
                close_games = team_matches.apply(
                    lambda row: abs(row['HomeGoals'] - row['AwayGoals']) == 1, 
                    axis=1
                ).sum()
                
                close_percent = 100 * close_games / len(team_matches)
                return round(avg_goals + (close_percent / 10), 2)
            except:
                return 0.0

        spannung_team1 = safe_calculate_team_spannung(team1, df)
        spannung_team2 = safe_calculate_team_spannung(team2, df)
        league_spannung_score = league_spannung.get(league, 0)
        league_draw_rate = league_zero_draw_rate.get(league, 0)

        # Calculate 0-0 draw stats
        goalless_matches = h2h_matches[(h2h_matches['HomeGoals'] == 0) & (h2h_matches['AwayGoals'] == 0)].shape[0]
        goalless_percentage = (goalless_matches / result['total_matches']) * 100 if result['total_matches'] > 0 else 0

        # Create summary card
        summary = dbc.Card([
            dbc.CardBody([
                html.H4(f"{team1} vs {team2}", className="card-title"),
                html.P(f"Total matches: {result['total_matches']}"),
                html.P(f"0-0 Draws in H2H: {goalless_matches} ({goalless_percentage:.2f}%)"),
                html.P(f"League 0-0 Avg: {league_draw_rate:.2f}%"),
                dbc.Row([
                    dbc.Col([
                        html.H5(f"{team1} Wins: {result['team1_wins']}"),
                        html.P(f"SpannungScore: {spannung_team1}")
                    ], width=4),
                    dbc.Col([
                        html.H5(f"Draws: {result['draws']}"),
                        html.P(f"League 0-0 rate: {league_draw_rate:.2f}%")
                    ], width=4),
                    dbc.Col([
                        html.H5(f"{team2} Wins: {result['team2_wins']}"),
                        html.P(f"SpannungScore: {spannung_team2}")
                    ], width=4)
                ]),
                html.P(f"League SpannungScore: {league_spannung_score:.2f}", className="mb-0")
            ])
        ], color="light")

        # Create pie chart
        fig = px.pie(
            names=[f'{team1} Wins', f'{team2} Wins', 'Draws'],
            values=[result.get('team1_wins', 0), 
                    result.get('team2_wins', 0), 
                    result.get('draws', 0)],
            title='Head-to-Head Results'
        )

        # Create line chart of goals over time
        line_data = h2h_matches.copy()
        line_data['Year'] = line_data['Date'].dt.year
        
        team1_data = line_data[((line_data['HomeTeam'] == team1) | (line_data['AwayTeam'] == team1))].copy()
        team1_data['TeamGoals'] = team1_data.apply(
            lambda row: row['HomeGoals'] if row['HomeTeam'] == team1 else row['AwayGoals'], axis=1)
        team1_avg = team1_data.groupby('Year')['TeamGoals'].mean().reset_index(name=team1)
        
        team2_data = line_data[((line_data['HomeTeam'] == team2) | (line_data['AwayTeam'] == team2))].copy()
        team2_data['TeamGoals'] = team2_data.apply(
            lambda row: row['HomeGoals'] if row['HomeTeam'] == team2 else row['AwayGoals'], axis=1)
        team2_avg = team2_data.groupby('Year')['TeamGoals'].mean().reset_index(name=team2)
        
        merged_avg = pd.merge(team1_avg, team2_avg, on='Year', how='outer').sort_values('Year')
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(
            x=merged_avg['Year'], 
            y=merged_avg[team1], 
            mode='lines+markers', 
            name=team1,
            line=dict(color='blue')
        ))
        line_fig.add_trace(go.Scatter(
            x=merged_avg['Year'], 
            y=merged_avg[team2], 
            mode='lines+markers', 
            name=team2,
            line=dict(color='red')
        ))
        line_fig.update_layout(
            title="Average Goals per Year by Team", 
            xaxis_title="Year", 
            yaxis_title="Goals",
            hovermode="x unified"
        )

        # Home/Away performance graph
        home_team1 = h2h_matches[h2h_matches['HomeTeam'] == team1]
        home_team2 = h2h_matches[h2h_matches['HomeTeam'] == team2]
        team1_home_wins = (home_team1['HomeGoals'] > home_team1['AwayGoals']).sum()
        team2_home_wins = (home_team2['HomeGoals'] > home_team2['AwayGoals']).sum()

        home_away_fig = go.Figure()
        home_away_fig.add_trace(go.Bar(
            x=[f'{team1} at Home', f'{team2} at Home'],
            y=[team1_home_wins, team2_home_wins],
            name='Wins',
            marker_color='green'
        ))
        home_away_fig.add_trace(go.Bar(
            x=[f'{team1} at Home', f'{team2} at Home'],
            y=[(home_team1.shape[0] - team1_home_wins), (home_team2.shape[0] - team2_home_wins)],
            name='Losses/Draws',
            marker_color='gray'
        ))
        home_away_fig.update_layout(
            title='Home Performance in H2H Matches',
            barmode='stack'
        )

        # Seasonal goals analysis
        h2h_matches.loc[:, 'Month'] = h2h_matches['Date'].dt.month
        monthly_goals = h2h_matches.groupby('Month')['TotalGoals'].mean().reset_index()

        seasonal_fig = px.bar(
            monthly_goals,
            x='Month',
            y='TotalGoals',
            title='Average Goals by Month in H2H Matches',
            labels={'TotalGoals': 'Avg. Goals', 'Month': 'Month'},
            color='TotalGoals',
            color_continuous_scale='Viridis'
        )
        seasonal_fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        # Additional insights
        impact = dbc.Alert([
            html.H5("Additional Context", className="alert-heading"),
            html.P(f"The {league} has a 0-0 draw rate of {league_draw_rate:.2f}%, which may influence the likelihood of a draw."),
            html.P(f"The league's overall excitement score is {league_spannung_score:.2f}, suggesting {'high' if league_spannung_score > 2 else 'moderate'} scoring potential."),
            html.P(f"{team1} has a higher excitement score than {team2}" if spannung_team1 > spannung_team2 
                  else f"{team2} has a higher excitement score than {team1}" if spannung_team2 > spannung_team1 
                  else "Both teams have similar excitement scores")
        ], color="secondary")

        return summary, fig, line_fig, impact, home_away_fig, seasonal_fig

    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}")
        return (
            dbc.Alert(f"Error processing data: {str(e)}", color="danger"),
            error_fig,
            error_fig,
            "",
            error_fig,
            error_fig
        )
# Run the app
# ------------------------
if __name__ == '__main__':
    app.run(debug=True)