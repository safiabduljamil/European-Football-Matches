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

# Data Loading and Preparation

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

# ------------------------
# Load and prepare all data
# ------------------------
matches_df = load_match_data()
league_stats = compute_league_stats(matches_df)
league_zero_draw_rate = league_stats.set_index('League')['GoallessPercentage'].to_dict()
league_spannung = league_stats.set_index('League')['SpannungsScore'].to_dict()
home_advantage_data = prepare_home_advantage_data(matches_df)
seasonal_data = prepare_seasonal_data(matches_df)

# ------------------------
# App1.py Data Processing
# ------------------------
def load_data_app1():
    """Load data for the extended home advantage analysis"""
    directory = '../shared-data'
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Folder '{directory}' not found.")

    data_frames = []
    for filename in os.listdir(directory):
        if filename.endswith(('.xlsx', '.xls', '.csv')):
            file_path = os.path.join(directory, filename)
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path, engine='openpyxl')
                data_frames.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df
    else:
        print("No valid files found in the folder.")
        return None

# Load the data for extended analysis
app1_df = load_data_app1()

# Data cleaning and preparation for app1
if app1_df is not None:
    app1_df['Date'] = pd.to_datetime(app1_df['Date'], dayfirst=True, errors='coerce')
    app1_df = app1_df.dropna(subset=['Date'])
    app1_df['Year'] = app1_df['Date'].dt.year
    app1_df['Result'] = app1_df.apply(
        lambda row: 'H' if row['HomeGoals'] > row['AwayGoals'] 
        else 'A' if row['HomeGoals'] < row['AwayGoals'] 
        else 'D', axis=1
    )

# Calculate basic statistics for app1
if app1_df is not None:
    total_matches_app1 = app1_df.shape[0]
    home_wins_app1 = (app1_df['Result'] == 'H').sum()
    away_wins_app1 = (app1_df['Result'] == 'A').sum()
    draws_app1 = (app1_df['Result'] == 'D').sum()

    home_win_pct_app1 = (home_wins_app1 / total_matches_app1) * 100
    away_win_pct_app1 = (away_wins_app1 / total_matches_app1) * 100
    draw_pct_app1 = (draws_app1 / total_matches_app1) * 100

    avg_home_goals_app1 = app1_df['HomeGoals'].mean()
    avg_away_goals_app1 = app1_df['AwayGoals'].mean()

    # Create yearly stats
    yearly_stats_app1 = app1_df.groupby('Year').agg(
        total_matches=('Result', 'count'),
        home_wins=('Result', lambda x: (x == 'H').sum()),
        away_wins=('Result', lambda x: (x == 'A').sum()),
        avg_home_goals=('HomeGoals', 'mean'),
        avg_away_goals=('AwayGoals', 'mean')
    )
    
    yearly_stats_app1['home_win_percentage'] = (yearly_stats_app1['home_wins'] / yearly_stats_app1['total_matches']) * 100
    yearly_stats_app1['away_win_percentage'] = (yearly_stats_app1['away_wins'] / yearly_stats_app1['total_matches']) * 100

    # Create figures for Tab 1
    match_fig_app1 = go.Figure([
        go.Bar(
            x=['Home Wins', 'Away Wins', 'Draws'],
            y=[home_win_pct_app1, away_win_pct_app1, draw_pct_app1],
            marker_color=['royalblue', 'darkorange', 'gold'],
            text=[f"{home_win_pct_app1:.1f}%", f"{away_win_pct_app1:.1f}%", f"{draw_pct_app1:.1f}%"],
            textposition='auto'
        )
    ])

    match_fig_app1.update_layout(
        title='Win Rate: Home vs Away Teams',
        yaxis_title='Percentage (%)',
        yaxis=dict(range=[0, 100])
    )

    goal_fig_app1 = go.Figure([
        go.Bar(
            x=['Home Team', 'Away Team'],
            y=[avg_home_goals_app1, avg_away_goals_app1],
            marker_color=['royalblue', 'darkorange'],
            text=[f"{avg_home_goals_app1:.2f}", f"{avg_away_goals_app1:.2f}"],
            textposition='auto'
        )
    ])

    goal_fig_app1.update_layout(
        title='Average Goals: Home vs Away Teams',
        yaxis_title='Average Goals Per Match'
    )

    # Create figures for Tab 2 (Trends)
    years_app1 = yearly_stats_app1.index
    home_win_pct_app1 = yearly_stats_app1['home_win_percentage']
    away_win_pct_app1 = yearly_stats_app1['away_win_percentage']
    home_goals_app1 = yearly_stats_app1['avg_home_goals']
    away_goals_app1 = yearly_stats_app1['avg_away_goals']

    # Win rate trend
    z_home_win = np.polyfit(years_app1, home_win_pct_app1, 1)
    p_home_win = np.poly1d(z_home_win)

    z_away_win = np.polyfit(years_app1, away_win_pct_app1, 1)
    p_away_win = np.poly1d(z_away_win)

    fig_winrate_trend_app1 = go.Figure()
    fig_winrate_trend_app1.add_trace(go.Scatter(x=years_app1, y=home_win_pct_app1, mode='lines+markers', name='Home Win %'))
    fig_winrate_trend_app1.add_trace(go.Scatter(x=years_app1, y=away_win_pct_app1, mode='lines+markers', name='Away Win %'))
    fig_winrate_trend_app1.add_trace(go.Scatter(x=years_app1, y=p_home_win(years_app1), mode='lines', line=dict(dash='dash', color='red'), name='Home Trend'))
    fig_winrate_trend_app1.add_trace(go.Scatter(x=years_app1, y=p_away_win(years_app1), mode='lines', line=dict(dash='dash', color='blue'), name='Away Trend'))
    fig_winrate_trend_app1.update_layout(
        title='Win Rate Trends: Home vs Away Teams',
        xaxis_title='Year',
        yaxis_title='Win Percentage (%)',
        template='plotly_white'
    )

    # Goals trend
    z_home_goals = np.polyfit(years_app1, home_goals_app1, 1)
    p_home_goals = np.poly1d(z_home_goals)

    z_away_goals = np.polyfit(years_app1, away_goals_app1, 1)
    p_away_goals = np.poly1d(z_away_goals)

    fig_goals_avg_app1 = go.Figure()
    fig_goals_avg_app1.add_trace(go.Scatter(x=years_app1, y=home_goals_app1, mode='lines+markers', name='Home Goals', marker=dict(symbol='circle')))
    fig_goals_avg_app1.add_trace(go.Scatter(x=years_app1, y=away_goals_app1, mode='lines+markers', name='Away Goals', marker=dict(symbol='square')))
    fig_goals_avg_app1.add_trace(go.Scatter(x=years_app1, y=p_home_goals(years_app1), mode='lines', line=dict(dash='dash', color='red'), name='Home Trend'))
    fig_goals_avg_app1.add_trace(go.Scatter(x=years_app1, y=p_away_goals(years_app1), mode='lines', line=dict(dash='dash', color='blue'), name='Away Trend'))
    fig_goals_avg_app1.update_layout(
        title='Average Goals Trends: Home vs Away',
        xaxis_title='Year',
        yaxis_title='Average Goals',
        legend=dict(x=0.01, y=0.99)
    )

    # Country-specific analysis
    country_stats_app1 = app1_df.groupby('League').agg(
        total_matches=('Result', 'count'),
        home_wins=('Result', lambda x: (x == 'H').sum()),
        away_wins=('Result', lambda x: (x == 'A').sum()),
        avg_home_goals=('HomeGoals', 'mean'),
        avg_away_goals=('AwayGoals', 'mean')
    )

    # Calculate percentages
    country_stats_app1['home_win_pct'] = (country_stats_app1['home_wins'] / country_stats_app1['total_matches']) * 100
    country_stats_app1['away_win_pct'] = (country_stats_app1['away_wins'] / country_stats_app1['total_matches']) * 100
    country_stats_app1['win_pct_diff'] = country_stats_app1['home_win_pct'] - country_stats_app1['away_win_pct']
    country_stats_app1['goal_diff'] = country_stats_app1['avg_home_goals'] - country_stats_app1['avg_away_goals']

    # Normalize values for scoring
    country_stats_app1['norm_win_diff'] = (country_stats_app1['win_pct_diff'] - country_stats_app1['win_pct_diff'].min()) / (country_stats_app1['win_pct_diff'].max() - country_stats_app1['win_pct_diff'].min())
    country_stats_app1['norm_goal_diff'] = (country_stats_app1['goal_diff'] - country_stats_app1['goal_diff'].min()) / (country_stats_app1['goal_diff'].max() - country_stats_app1['goal_diff'].min())
    country_stats_app1['total_score'] = (country_stats_app1['norm_win_diff'] * 0.5) + (country_stats_app1['norm_goal_diff'] * 0.5)

    # Sort for visualization
    country_stats_sorted_win_app1 = country_stats_app1.sort_values('home_win_pct', ascending=False)
    country_stats_sorted_win_diff_app1 = country_stats_app1.sort_values('win_pct_diff', ascending=False)
    country_stats_sorted_goal_app1 = country_stats_app1.sort_values('avg_home_goals', ascending=False)
    country_stats_sorted_goal_diff_app1 = country_stats_app1.sort_values('goal_diff', ascending=False)
    country_stats_sorted_score_app1 = country_stats_app1.sort_values('total_score', ascending=False)

    # Country trend analysis
    country_yearly_stats_app1 = app1_df.groupby(['League', 'Year']).agg(
        total_matches=('Result', 'count'),
        home_wins=('Result', lambda x: (x == 'H').sum()),
        away_wins=('Result', lambda x: (x == 'A').sum()),
        avg_home_goals=('HomeGoals', 'mean'),
        avg_away_goals=('AwayGoals', 'mean')
    ).reset_index()

    # Calculate percentages and differences
    country_yearly_stats_app1['home_win_pct'] = (country_yearly_stats_app1['home_wins'] / country_yearly_stats_app1['total_matches']) * 100
    country_yearly_stats_app1['away_win_pct'] = (country_yearly_stats_app1['away_wins'] / country_yearly_stats_app1['total_matches']) * 100
    country_yearly_stats_app1['win_pct_diff'] = country_yearly_stats_app1['home_win_pct'] - country_yearly_stats_app1['away_win_pct']
    country_yearly_stats_app1['goal_diff'] = country_yearly_stats_app1['avg_home_goals'] - country_yearly_stats_app1['avg_away_goals']

    # Calculate trend slopes for each country
    trend_data = []
    for league in country_yearly_stats_app1['League'].unique():
        league_data = country_yearly_stats_app1[country_yearly_stats_app1['League'] == league]
        
        # Win percentage difference trend
        z_win = np.polyfit(league_data['Year'], league_data['win_pct_diff'], 1)
        slope_win = z_win[0]
        
        # Goal difference trend
        z_goal = np.polyfit(league_data['Year'], league_data['goal_diff'], 1)
        slope_goal = z_goal[0]
        
        trend_data.append({
            'League': league,
            'win_diff_slope': slope_win,
            'goal_diff_slope': slope_goal
        })

    trend_df = pd.DataFrame(trend_data)

    # Normalize trend slopes
    trend_df['norm_win_slope'] = (trend_df['win_diff_slope'] - trend_df['win_diff_slope'].min()) / (trend_df['win_diff_slope'].max() - trend_df['win_diff_slope'].min())
    trend_df['norm_goal_slope'] = (trend_df['goal_diff_slope'] - trend_df['goal_diff_slope'].min()) / (trend_df['goal_diff_slope'].max() - trend_df['goal_diff_slope'].min())
    trend_df['total_trend_score'] = (trend_df['norm_win_slope'] * 0.5) + (trend_df['norm_goal_slope'] * 0.5)

    # Sort for visualization
    trend_df_sorted_win = trend_df.sort_values('win_diff_slope')
    trend_df_sorted_goal = trend_df.sort_values('goal_diff_slope')
    trend_df_sorted_score = trend_df.sort_values('total_trend_score')

    # Precompute figures for Tab 5
    fig_tab3_11 = go.Figure(go.Bar(
        x=country_stats_sorted_score_app1.index,
        y=country_stats_sorted_score_app1['total_score'],
        marker_color='blue'
    ))
    fig_tab3_11.update_layout(
        title='Home Advantage Score by Country',
        xaxis_title='Country',
        yaxis_title='Total Score (0-1)'
    )

    fig_tab4_18 = go.Figure(go.Bar(
        x=trend_df_sorted_score['League'],
        y=trend_df_sorted_score['total_trend_score'],
        marker_color='green'
    ))
    fig_tab4_18.update_layout(
        title='Home Advantage Decline Score',
        xaxis_title='Country',
        yaxis_title='Total Trend Score (0-1)'
    )

    # Create combined score for Tab 5
    df_tab5 = country_stats_app1[['total_score']].reset_index()
    df_tab5 = df_tab5.merge(trend_df[['League', 'total_trend_score']], on='League', how='inner')
    df_tab5['combined_score'] = df_tab5['total_score'] - df_tab5['total_trend_score']
    df_tab5 = df_tab5.sort_values('combined_score', ascending=False)

    fig_combined_score = go.Figure(go.Bar(
        x=df_tab5['League'],
        y=df_tab5['combined_score'],
        marker_color='purple'
    ))
    fig_combined_score.update_layout(
        title='Combined Home Advantage Score',
        xaxis_title='Country',
        yaxis_title='Combined Score'
    )

# ------------------------
# App Setup
# ------------------------
app = Dash(__name__, 
           external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/bWLwgP.css'], 
           suppress_callback_exceptions=True)

app.title = "European Football Statistics Dashboard"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("European Football Statistics Dashboard", 
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
                
                # Tab 2: Home Advantage Analysis (Extended from app1.py)
                dcc.Tab(label="Home Advantage Analysis", value='tab-home-extended', className='custom-tab', children=[
                    html.Br(),
                    html.Div([
                        html.H1("HOME ADVANTAGE ANALYSIS", style={'textAlign': 'center'}),
                        html.H2("Does home advantage exist in European football leagues, and has it decreased over the years?", 
                                style={'textAlign': 'center'}),
                        html.P("This analysis is based on a dataset of 216,883 matches from 22 leagues in 11 countries between 1993 and 2023. "
                               "Through five thematic focuses, 19 different visualizations were created and various insights were gained.",
                               style={'textAlign': 'center'}),
                        dcc.Tabs(id="app1-tabs", value='app1-tab1', children=[
                            # TAB 1: Home Advantage
                            dcc.Tab(label='Home Advantage', value='app1-tab1', children=[
                                html.Div([
                                    html.H3("Analysis of home advantage in European football"),
                                    html.P("Figure 1 compares the win rates of home and away teams across all matches."),
                                    html.P("Figure 2 shows the comparison of average goals per match between home and away teams."),
                                    html.P("The result: With a win rate of 45% and an average of 1.50 goals per match, home teams perform significantly better than away teams. "
                                           "This confirms the well-known home advantage in football."),
                                    
                                    dcc.Dropdown(
                                        id='app1-tab1-dropdown',
                                        options=[
                                            {'label': 'Figure 1: Win Rate', 'value': 'win_rate'},
                                            {'label': 'Figure 2: Average Goals', 'value': 'avg_goals'}
                                        ],
                                        value='win_rate'
                                    ),
                                    dcc.Graph(id='app1-tab1-graph')
                                ], style={'padding': '20px'})
                            ]),

                            # TAB 2: Development of Home Advantage
                            dcc.Tab(label='Development Over Time', value='app1-tab2', children=[
                                html.Div([
                                    html.H3("Analysis of the development of home advantage in European football over time"),
                                    html.P("Figure 3 shows the change in win rates of home and away teams over the years using trend lines."),
                                    html.P("Figure 4 shows the change in average goals scored by home and away teams over the years using trend lines."),
                                    html.P("Both win rates and average goals show a decline for home teams over the years. In contrast, away teams show a steady upward trend. "
                                           "Despite smaller fluctuations over time, home teams overall win less often and score fewer goals, "
                                           "while away teams are becoming more successful and score more frequently."),
                                    html.P("An extraordinary influence outside the main topic is visible in this section. "
                                           "In both figures, a significant negative deviation from the average is visible in 2020. "
                                           "This phase lasts about a year before the values recover. This pattern strongly resembles the COVID-19 pandemic. "
                                           "During that time, when games were played without spectators and strict restrictions were in place, "
                                           "home advantage reached its lowest level. This suggests that the presence of fans has a noticeable impact on home advantage."),
                                    
                                    dcc.Dropdown(
                                        id='app1-tab2-dropdown',
                                        options=[
                                            {'label': 'Figure 3: Win Rate Trend', 'value': 'win_rate_trend'},
                                            {'label': 'Figure 4: Goals Trend', 'value': 'goals_trend'}
                                        ],
                                        value='win_rate_trend'
                                    ),
                                    dcc.Graph(id='app1-tab2-graph')
                                ], style={'padding': '20px'})
                            ]),

                            # TAB 3: Home Advantage by Country
                            dcc.Tab(label='By Country', value='app1-tab3', children=[
                                html.Div([
                                    html.H3("Analysis of home advantage in European football by country"),
                                    html.P("Figure 5 shows the win rates of home and away teams by country. Countries are sorted in descending order by home win rate. "
                                           "However, this order is not completely meaningful, as the sometimes strong fluctuations in away win rates are noticeable. "
                                           "For a more realistic assessment, not only a high home win rate is relevant, but also as low an away win rate as possible."),
                                    html.P("Therefore, Figure 6 calculates and visualizes the difference between home and away win rates. "
                                           "This results in a more meaningful ranking - for example, France moves from 8th to 2nd place."),
                                    html.P("In Figure 7, these difference values are normalized to a range of 0 to 1 and converted into a score for further analysis."),
                                    html.P("Figure 8 shows the average goals of home and away teams by country. Sorting is again descending by home goals. "
                                           "This order is not very meaningful in terms of home advantage, as it only shows in which countries many goals are scored in general - such as in the Netherlands."),
                                    html.P("To better assess the performance of home teams relative to away teams, Figure 9 calculates the goal difference and uses it as the basis for the ranking. "
                                           "This highlights countries where home teams are particularly successful. For example, Greece rises from 5th place in Figure 8 to 1st place in Figure 9, "
                                           "while Scotland falls from 5th to last place."),
                                    html.P("In Figure 10, these goal difference values are normalized to a range of 0 to 1 for further processing."),
                                    html.P("Figure 11 combines the results of Figures 7 and 10 to evaluate home advantage by country. "
                                           "The normalized values of the win rate difference and the goal difference were each weighted at 50% and combined into an overall value."),
                                    html.P("The result: Greece, which ranks first in both individual figures, clearly leads the ranking. "
                                           "England (10th place) and Scotland (11th place) retain their positions in both tables. "
                                           "The remaining countries swap places with each other and are ranked in the overall ranking based on small score differences."),
                                    
                                    dcc.Dropdown(
                                        id='app1-tab3-dropdown',
                                        options=[
                                            {'label': 'Figure 5: Win Rate by Country', 'value': 'win_rate_country'},
                                            {'label': 'Figure 6: Win Rate Difference', 'value': 'win_rate_diff'},
                                            {'label': 'Figure 7: Normalized Win Rate Difference', 'value': 'norm_win_diff'},
                                            {'label': 'Figure 8: Average Goals by Country', 'value': 'avg_goals_country'},
                                            {'label': 'Figure 9: Goal Difference', 'value': 'goal_diff'},
                                            {'label': 'Figure 10: Normalized Goal Difference', 'value': 'norm_goal_diff'},
                                            {'label': 'Figure 11: Overall Home Advantage Score', 'value': 'overall_score'}
                                        ],
                                        value='win_rate_country'
                                    ),
                                    dcc.Graph(id='app1-tab3-graph')
                                ], style={'padding': '20px'})
                            ]),

                            # TAB 4: Development by Country
                            dcc.Tab(label='Development by Country', value='app1-tab4', children=[
                                html.Div([
                                    html.H3("Analysis of the development of home advantage in European football by country over time"),
                                    html.P("Figure 12 directly examines the difference in win rates between home and away teams - based on the findings from the previous analysis. "
                                           "For each country, this difference is calculated and its development over the years is shown. "
                                           "Due to the large number of lines, the presentation initially appears complex, but is intended for readers who want to analyze individual countries in detail."),
                                    html.P("To enable a cross-country comparison, Figure 13 calculates the trend slope of this difference for each country. "
                                           "This shows how home advantage has changed over time. All countries show a negative development - meaning that home advantage is declining everywhere. "
                                           "The smallest change is observed in Scotland, while France shows the strongest decline."),
                                    html.P("In Figure 14, the trend slope values are normalized to a range of 0 to 1 to calculate standardized points. "
                                           "These points will later be used to form an overall score."),
                                    html.P("Figure 15 calculates the difference in average goals between home and away teams and shows its development over the years. "
                                           "As in Figure 12, the result is a very confusing picture - this representation is therefore particularly suitable for examining individual countries."),
                                    html.P("For better comparison, Figure 16 calculates the trend slope of the goal difference per country and ranks them. "
                                           "Here too, all countries show a declining trend. The strongest decline is recorded in Greece, "
                                           "while the smallest change is again observed in Scotland."),
                                    html.P("To reuse these values, they are normalized to a range of 0 to 1 in Figure 17 and converted into standardized point values."),
                                    html.P("Figure 18 combines the previously normalized values from Figure 14 and Figure 17 - that is, the point values of the win rate and goal difference - "
                                           "each weighted at 50% and combined into an overall score to evaluate the decline in home advantage over time."),
                                    html.P("The result: Greece shows the strongest decline in home advantage over the years. "
                                           "Scotland, which ranks last in both individual evaluations, shows the least change as expected. "
                                           "The development of the other countries can be seen in detail in the figure."),
                                    
                                    dcc.Dropdown(
                                        id='app1-tab4-dropdown',
                                        options=[
                                            {'label': 'Figure 12: Win Rate Difference Over Time', 'value': 'win_diff_trend'},
                                            {'label': 'Figure 13: Trend Slope of Win Rate Difference', 'value': 'win_diff_slope'},
                                            {'label': 'Figure 14: Normalized Win Difference Slope', 'value': 'norm_win_slope'},
                                            {'label': 'Figure 15: Goal Difference Over Time', 'value': 'goal_diff_trend'},
                                            {'label': 'Figure 16: Trend Slope of Goal Difference', 'value': 'goal_diff_slope'},
                                            {'label': 'Figure 17: Normalized Goal Difference Slope', 'value': 'norm_goal_slope'},
                                            {'label': 'Figure 18: Overall Decline Score', 'value': 'overall_decline_score'}
                                        ],
                                        value='win_diff_trend'
                                    ),
                                    dcc.Graph(id='app1-tab4-graph')
                                ], style={'padding': '20px'})
                            ]),

                            # TAB 5: Additional Analysis
                            dcc.Tab(label='Additional Analysis', value='app1-tab5', children=[
                                html.Div([
                                    html.H3("Linked analysis of home advantage by country and trend"),
                                    html.P("In this section, an analysis is conducted that combines the point values from Figure 11 (home advantage by country) "
                                           "with the change values from Figure 17 (decline of home advantage over the years). "
                                           "Figure 11 and Figure 17 are shown side by side below."),
                                    html.P("For each country, the point value from Figure 17 is subtracted from the point value from Figure 11 to obtain a new overall value. "
                                           "Based on this, a new ranking is created in Figure 19."),
                                    html.P("The goal of this final presentation is to show how strong the home advantage was originally in a country, "
                                           "how much it has declined over the years, and where the respective country currently stands in comparison."),
                                    html.P("An example of this is Greece and Scotland: While Greece originally had the highest home advantage, "
                                           "it simultaneously shows the strongest decline. Scotland, on the other hand, has the least home advantage, "
                                           "but shows almost no change over the years. As a result, both countries end up in the middle of the combined ranking."),
                                    html.P("Looking at Spain, the country ranks 4th with a home advantage value of 0.7. At the same time, "
                                           "it is one of the countries with the least decline (2nd place). This means: Spain manages to largely maintain its existing home advantage - "
                                           "which suggests that the country has the potential to rise to the top of the ranking in the coming years."),
                                    html.P("Looking at the Netherlands, the country ranks 2nd in home advantage and only third from last in its decline. "
                                           "This means that the Netherlands have largely preserved their home advantage and rank second in the combined ranking."),
                                    html.P("Belgium, Germany and Turkey also belong to the countries that have at least partially preserved their home advantage. "
                                           "They occupy places 3, 4 and 5 in the ranking of stability."),
                                    html.P("Portugal, on the other hand, forms the tail end of this analysis. The country ranks only 9th in home advantage and simultaneously shows the third strongest decline. "
                                           "This suggests that Portugal cannot maintain its home advantage in the long term and is moving towards the last place."),
                                    html.P("England, France and Italy also belong to the countries where home advantage is declining particularly quickly."),
                                
                                    html.Div([
                                        html.Div(dcc.Graph(figure=fig_tab3_11), style={'width': '48%', 'display': 'inline-block'}),
                                        html.Div(dcc.Graph(figure=fig_tab4_18), style={'width': '48%', 'display': 'inline-block'})
                                    ], style={'margin-bottom': '20px'}),
                                    
                                    html.H4("Combined Home Advantage Score (Home Advantage Score - Decline Score)", style={'textAlign': 'center'}),
                                    dcc.Graph(figure=fig_combined_score)
                                ], style={'padding': '20px'})
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
# Callbacks for app1.py functionality
# ------------------------

# Callback for Tab-1
@app.callback(
    Output('app1-tab1-graph', 'figure'),
    Input('app1-tab1-dropdown', 'value')
)
def update_tab1_graph(selected_value):
    if selected_value == 'win_rate':
        return match_fig_app1
    elif selected_value == 'avg_goals':
        return goal_fig_app1
    return match_fig_app1  # Default

# Callback for Tab-2
@app.callback(
    Output('app1-tab2-graph', 'figure'),
    Input('app1-tab2-dropdown', 'value')
)
def update_tab2_graph(selected_value):
    if selected_value == 'win_rate_trend':
        return fig_winrate_trend_app1
    elif selected_value == 'goals_trend':
        return fig_goals_avg_app1
    return fig_winrate_trend_app1  # Default

# Callback for Tab-3
@app.callback(
    Output('app1-tab3-graph', 'figure'),
    Input('app1-tab3-dropdown', 'value')
)
def update_tab3_graph(selected_value):
    if selected_value == 'win_rate_country':
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=country_stats_sorted_win_app1.index,
            y=country_stats_sorted_win_app1['home_win_pct'],
            name='Home Win Rate',
            marker_color='royalblue'
        ))
        fig.add_trace(go.Bar(
            x=country_stats_sorted_win_app1.index,
            y=country_stats_sorted_win_app1['away_win_pct'],
            name='Away Win Rate',
            marker_color='darkorange'
        ))
        fig.update_layout(
            title='Win Rates by Country (Sorted by Home Win Rate)',
            xaxis_title='Country',
            yaxis_title='Win Rate (%)',
            barmode='group'
        )
        return fig
        
    elif selected_value == 'win_rate_diff':
        fig = go.Figure(go.Bar(
            x=country_stats_sorted_win_diff_app1.index,
            y=country_stats_sorted_win_diff_app1['win_pct_diff'],
            marker_color='green'
        ))
        fig.update_layout(
            title='Home vs Away Win Rate Difference',
            xaxis_title='Country',
            yaxis_title='Difference (%)'
        )
        return fig
        
    elif selected_value == 'norm_win_diff':
        fig = go.Figure(go.Bar(
            x=country_stats_sorted_win_diff_app1.index,
            y=country_stats_sorted_win_diff_app1['norm_win_diff'],
            marker_color='purple'
        ))
        fig.update_layout(
            title='Normalized Win Rate Difference',
            xaxis_title='Country',
            yaxis_title='Normalized Value (0-1)'
        )
        return fig
        
    elif selected_value == 'avg_goals_country':
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=country_stats_sorted_goal_app1.index,
            y=country_stats_sorted_goal_app1['avg_home_goals'],
            name='Home Goals',
            marker_color='royalblue'
        ))
        fig.add_trace(go.Bar(
            x=country_stats_sorted_goal_app1.index,
            y=country_stats_sorted_goal_app1['avg_away_goals'],
            name='Away Goals',
            marker_color='darkorange'
        ))
        fig.update_layout(
            title='Average Goals by Country (Sorted by Home Goals)',
            xaxis_title='Country',
            yaxis_title='Average Goals per Match',
            barmode='group'
        )
        return fig
        
    elif selected_value == 'goal_diff':
        fig = go.Figure(go.Bar(
            x=country_stats_sorted_goal_diff_app1.index,
            y=country_stats_sorted_goal_diff_app1['goal_diff'],
            marker_color='red'
        ))
        fig.update_layout(
            title='Home vs Away Goal Difference',
            xaxis_title='Country',
            yaxis_title='Goal Difference'
        )
        return fig
        
    elif selected_value == 'norm_goal_diff':
        fig = go.Figure(go.Bar(
            x=country_stats_sorted_goal_diff_app1.index,
            y=country_stats_sorted_goal_diff_app1['norm_goal_diff'],
            marker_color='orange'
        ))
        fig.update_layout(
            title='Normalized Goal Difference',
            xaxis_title='Country',
            yaxis_title='Normalized Value (0-1)'
        )
        return fig
        
    elif selected_value == 'overall_score':
        fig = go.Figure(go.Bar(
            x=country_stats_sorted_score_app1.index,
            y=country_stats_sorted_score_app1['total_score'],
            marker_color='blue'
        ))
        fig.update_layout(
            title='Overall Home Advantage Score by Country',
            xaxis_title='Country',
            yaxis_title='Total Score (0-1)'
        )
        return fig

# Callback for Tab-4
@app.callback(
    Output('app1-tab4-graph', 'figure'),
    Input('app1-tab4-dropdown', 'value')
)
def update_tab4_graph(selected_value):
    if selected_value == 'win_diff_trend':
        fig = go.Figure()
        for league in country_yearly_stats_app1['League'].unique():
            league_data = country_yearly_stats_app1[country_yearly_stats_app1['League'] == league]
            fig.add_trace(go.Scatter(
                x=league_data['Year'],
                y=league_data['win_pct_diff'],
                mode='lines+markers',
                name=league
            ))
        fig.update_layout(
            title='Win Rate Difference Development by Country',
            xaxis_title='Year',
            yaxis_title='Home vs Away Win Rate Difference (%)'
        )
        return fig
        
    elif selected_value == 'win_diff_slope':
        fig = go.Figure(go.Bar(
            x=trend_df_sorted_win['League'],
            y=trend_df_sorted_win['win_diff_slope'],
            marker_color='red'
        ))
        fig.update_layout(
            title='Trend of Win Rate Difference (More negative = stronger decline)',
            xaxis_title='Country',
            yaxis_title='Trend Slope'
        )
        return fig
        
    elif selected_value == 'norm_win_slope':
        fig = go.Figure(go.Bar(
            x=trend_df_sorted_win['League'],
            y=trend_df_sorted_win['norm_win_slope'],
            marker_color='purple'
        ))
        fig.update_layout(
            title='Normalized Trend Slope of Win Rate Difference',
            xaxis_title='Country',
            yaxis_title='Normalized Value (0-1)'
        )
        return fig
        
    elif selected_value == 'goal_diff_trend':
        fig = go.Figure()
        for league in country_yearly_stats_app1['League'].unique():
            league_data = country_yearly_stats_app1[country_yearly_stats_app1['League'] == league]
            fig.add_trace(go.Scatter(
                x=league_data['Year'],
                y=league_data['goal_diff'],
                mode='lines+markers',
                name=league
            ))
        fig.update_layout(
            title='Goal Difference Development by Country',
            xaxis_title='Year',
            yaxis_title='Home vs Away Goal Difference'
        )
        return fig
        
    elif selected_value == 'goal_diff_slope':
        fig = go.Figure(go.Bar(
            x=trend_df_sorted_goal['League'],
            y=trend_df_sorted_goal['goal_diff_slope'],
            marker_color='blue'
        ))
        fig.update_layout(
            title='Trend of Goal Difference (More negative = stronger decline)',
            xaxis_title='Country',
            yaxis_title='Trend Slope'
        )
        return fig
        
    elif selected_value == 'norm_goal_slope':
        fig = go.Figure(go.Bar(
            x=trend_df_sorted_goal['League'],
            y=trend_df_sorted_goal['norm_goal_slope'],
            marker_color='orange'
        ))
        fig.update_layout(
            title='Normalized Trend Slope of Goal Difference',
            xaxis_title='Country',
            yaxis_title='Normalized Value (0-1)'
        )
        return fig
        
    elif selected_value == 'overall_decline_score':
        fig = go.Figure(go.Bar(
            x=trend_df_sorted_score['League'],
            y=trend_df_sorted_score['total_trend_score'],
            marker_color='green'
        ))
        fig.update_layout(
            title='Overall Home Advantage Decline Score',
            xaxis_title='Country',
            yaxis_title='Total Trend Score (0-1)'
        )
        return fig

# ------------------------
# Existing Callbacks from app.py
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