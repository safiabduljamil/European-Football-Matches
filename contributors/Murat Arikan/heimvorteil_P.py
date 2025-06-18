import pandas as pd
import os
import matplotlib.pyplot as plt

# Directory containing the Excel files
directory = 'c:/Users/arikanmurat/dev/European-Football-Matches/shared-data'

# List to hold data from each file
data_frames = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.xlsx') or filename.endswith('.xls') or filename.endswith('csv'):
        file_path = os.path.join(directory, filename)
        # Read the Excel file
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, engine='openpyxl')
        # Append the DataFrame to the list
        data_frames.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(data_frames, ignore_index=True)

# Display the combined DataFrame
print(combined_df)


# Check the shape of the dataset (rows, columns)
print("Shape:", combined_df.shape)

# Show column names and types
print("\nColumn Info:")
print(combined_df.dtypes)

# Show first few rows
print("\nPreview:")
print(combined_df.head())

# Check for missing values
print("\nMissing values per column:")
print(combined_df.isnull().sum())

# Check unique values for 'League' and 'Result'
print("\nUnique leagues:", combined_df['League'].unique())
print("Unique match results:", combined_df['Result'].unique())

# Check for duplicate rows in the dataset
print("Duplicate rows:", combined_df.duplicated().sum())

# Check if 'Date' column was correctly parsed to datetime format
print("\nDate column preview:")
print(combined_df['Date'].head())
print("Date column type:", combined_df['Date'].dtype)

# Check the unique values in the 'Result' column
print("\nMatch result distribution:")
print(combined_df['Result'].value_counts())

# Calculate total number of matches
total_matches = combined_df.shape[0]

# Count match outcomes
home_wins = (combined_df['Result'] == 'H').sum()
away_wins = (combined_df['Result'] == 'A').sum()
draws = (combined_df['Result'] == 'D').sum()

# Calculate percentages
home_win_pct = (home_wins / total_matches) * 100
away_win_pct = (away_wins / total_matches) * 100
draw_pct = (draws / total_matches) * 100

# Display the results
print(f"Total matches: {total_matches}")
print(f"Home wins: {home_wins} ({home_win_pct:.2f}%)")
print(f"Away wins: {away_wins} ({away_win_pct:.2f}%)")
print(f"Draws: {draws} ({draw_pct:.2f}%)")

import plotly.graph_objs as go

match_fig = go.Figure([
    go.Bar(
        x=['Home Wins', 'Away Wins', 'Draws'],
        y=[45.2, 27.9, 26.8],
        marker_color=['royalblue', 'darkorange', 'gold'],
        text=["45.2%", "27.9%", "26.8%"],
        textposition='auto'
    )
])

match_fig.update_layout(
    title='Win Rate: Home vs Away Teams',
    yaxis_title='Percentage (%)',
    yaxis=dict(range=[0, 100])
)

goal_fig = go.Figure([
    go.Bar(
        x=['Home Team', 'Away Team'],
        y=[1.50, 1.13],
        marker_color=['royalblue', 'darkorange'],
        text=["1.50", "1.13"],
        textposition='auto'
    )
])

goal_fig.update_layout(
    title='Average Goals: Home vs Away Teams',
    yaxis_title='Average Goals Per Match'
)


# Convert 'Date' column to datetime format (if not already done)
combined_df['Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True)

# Extract 'Year' from 'Date'
combined_df['Year'] = combined_df['Date'].dt.year

# Extract the year from the datetime column
combined_df['Year'] = combined_df['Date'].dt.year

# Create full yearly_stats with all required columns
yearly_stats = combined_df.groupby('Year').agg(
    total_matches=('Result', 'count'),
    home_wins=('Result', lambda x: (x == 'H').sum()),
    away_wins=('Result', lambda x: (x == 'A').sum()),
    avg_home_goals=('HomeGoals', 'mean'),
    avg_away_goals=('AwayGoals', 'mean')
)

yearly_stats['home_win_percentage'] = (yearly_stats['home_wins'] / yearly_stats['total_matches']) * 100
yearly_stats['away_win_percentage'] = (yearly_stats['away_wins'] / yearly_stats['total_matches']) * 100

# Show column names to verify
print(yearly_stats.columns)




import plotly.graph_objs as go
import numpy as np

years = yearly_stats.index
home = yearly_stats['home_win_percentage']
away = yearly_stats['away_win_percentage']

# Trend lines
z_home = np.polyfit(years, home, 1)
p_home = np.poly1d(z_home)

z_away = np.polyfit(years, away, 1)
p_away = np.poly1d(z_away)

# Figure
fig_winrate_trend = go.Figure()

# Home actual
fig_winrate_trend.add_trace(go.Scatter(x=years, y=home, mode='lines+markers', name='Home Win %'))

# Away actual
fig_winrate_trend.add_trace(go.Scatter(x=years, y=away, mode='lines+markers', name='Away Win %'))

# Home trend line
fig_winrate_trend.add_trace(go.Scatter(x=years, y=p_home(years), mode='lines', line=dict(dash='dash', color='red'), name='Home Trend'))

# Away trend line
fig_winrate_trend.add_trace(go.Scatter(x=years, y=p_away(years), mode='lines', line=dict(dash='dash', color='blue'), name='Away Trend'))

# Layout
fig_winrate_trend.update_layout(
    title='Win Rate: Home vs Away Teams Over the Years',
    xaxis_title='Year',
    yaxis_title='Win Percentage (%)',
    template='plotly_white'
)



import numpy as np
import plotly.graph_objs as go

# Group the data by year and calculate average goals for home and away teams
yearly_stats = combined_df.groupby('Year').agg(
    avg_home_goals=('HomeGoals', 'mean'),
    avg_away_goals=('AwayGoals', 'mean')
)

# Extract years and goal averages
years = yearly_stats.index
home_goals = yearly_stats['avg_home_goals']
away_goals = yearly_stats['avg_away_goals']

# Calculate trend lines for home and away goals
z_home = np.polyfit(years, home_goals, 1)
p_home = np.poly1d(z_home)

z_away = np.polyfit(years, away_goals, 1)
p_away = np.poly1d(z_away)

# Create a Plotly figure
fig_goals_avg = go.Figure()

# Add line for average home goals
fig_goals_avg.add_trace(go.Scatter(
    x=years,
    y=home_goals,
    mode='lines+markers',
    name='Home Goals',
    marker=dict(symbol='circle')
))

# Add line for average away goals
fig_goals_avg.add_trace(go.Scatter(
    x=years,
    y=away_goals,
    mode='lines+markers',
    name='Away Goals',
    marker=dict(symbol='square')
))

# Add trend line for home goals
fig_goals_avg.add_trace(go.Scatter(
    x=years,
    y=p_home(years),
    mode='lines',
    name='Home Trend',
    line=dict(dash='dash', color='red')
))

# Add trend line for away goals
fig_goals_avg.add_trace(go.Scatter(
    x=years,
    y=p_away(years),
    mode='lines',
    name='Away Trend',
    line=dict(dash='dash', color='blue')
))

# Update layout settings
fig_goals_avg.update_layout(
    title='Average Goals: Home vs Away Over the Years',
    xaxis_title='Year',
    yaxis_title='Average Goals',
    template='plotly_white',
    legend=dict(x=0.01, y=0.99)
)

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

# Add a new column to identify the country of each match
combined_df['Country'] = combined_df['League'].map(league_to_country)

# Let's see the column we added
combined_df.head()

# Count match results per country
country_results = combined_df.groupby('Country')['Result'].value_counts().unstack().fillna(0)

# Optional: sort countries alphabetically
country_results = country_results.sort_index()

# Group by country and calculate counts
country_stats = combined_df.groupby('Country').agg(
    total_matches=('Result', 'count'),
    home_wins=('Result', lambda x: (x == 'H').sum()),
    away_wins=('Result', lambda x: (x == 'A').sum()),
    draws=('Result', lambda x: (x == 'D').sum())
)

# Calculate percentages
country_stats['home_win_pct'] = (country_stats['home_wins'] / country_stats['total_matches']) * 100
country_stats['away_win_pct'] = (country_stats['away_wins'] / country_stats['total_matches']) * 100
country_stats['draw_pct'] = (country_stats['draws'] / country_stats['total_matches']) * 100

# Sort by home_win_pct descending
country_stats_sorted = country_stats.sort_values(by='home_win_pct', ascending=False)

# Display rounded version
print(country_stats_sorted[['home_win_pct', 'away_win_pct', 'draw_pct']].round(2))

import plotly.graph_objs as go

# Sort by home win percentage descending
sorted_win_stats = country_stats.sort_values(by='home_win_pct', ascending=False)

# Create grouped bar chart
win_bar_fig = go.Figure()

win_bar_fig.add_trace(go.Bar(
    x=sorted_win_stats.index,
    y=sorted_win_stats['home_win_pct'],
    name='Home Win %',
    marker_color='royalblue',
    text=sorted_win_stats['home_win_pct'].round(2),
    texttemplate='%{text:.2f}%',
    textposition='outside'
))

win_bar_fig.add_trace(go.Bar(
    x=sorted_win_stats.index,
    y=sorted_win_stats['away_win_pct'],
    name='Away Win %',
    marker_color='darkorange',
    text=sorted_win_stats['away_win_pct'].round(2),
    texttemplate='%{text:.2f}%',
    textposition='outside'
))

# Step 3: Update layout
win_bar_fig.update_layout(
    barmode='group',
    title='Win Rate: Home vs Away by Country',
    xaxis_title='Country',
    yaxis_title='Win Percentage (%)',
    xaxis=dict(categoryorder='array', categoryarray=sorted_win_stats.index),
    yaxis=dict(range=[0, 60]),
    template='plotly_white'
)

# Calculate home vs away win percentage difference
country_stats['home_advantage_gap'] = country_stats['home_win_pct'] - country_stats['away_win_pct']

# Sort by home advantage gap descending
sorted_gap_stats = country_stats.sort_values(by='home_advantage_gap', ascending=False)


import plotly.graph_objs as go

# Create bar chart of home advantage gaps
win_gap_fig = go.Figure()

win_gap_fig.add_trace(go.Bar(
    x=sorted_gap_stats.index,
    y=sorted_gap_stats['home_advantage_gap'].round(2),
    marker_color='seagreen',
    text=sorted_gap_stats['home_advantage_gap'].round(2),
    texttemplate='%{text:.2f}%',
    textposition='outside',
    name='Home Advantage Gap (%)'
))

# Layout ayarları
win_gap_fig.update_layout(
    title='Win Rate Gap by Country',
    xaxis_title='Country',
    yaxis_title='Advantage Gap (%)',
    yaxis=dict(range=[0, max(sorted_gap_stats['home_advantage_gap']) + 5]),
    xaxis=dict(categoryorder='array', categoryarray=sorted_gap_stats.index),
    template='plotly_white'
)


# Normalize home advantage gap
country_stats['normalized_gap'] = (
    (country_stats['home_win_pct'] - country_stats['away_win_pct']) - 
    (country_stats['home_win_pct'] - country_stats['away_win_pct']).min()
) / (
    (country_stats['home_win_pct'] - country_stats['away_win_pct']).max() - 
    (country_stats['home_win_pct'] - country_stats['away_win_pct']).min()
)

# Yuvarla ve sırala
country_stats['normalized_gap'] = country_stats['normalized_gap'].round(2)
gap_sorted = country_stats.sort_values(by='normalized_gap', ascending=False)


import plotly.graph_objs as go

# Build the bar chart
fig_winrate_gap_normalized = go.Figure()

fig_winrate_gap_normalized.add_trace(go.Bar(
    x=gap_sorted.index,
    y=gap_sorted['normalized_gap'],
    text=gap_sorted['normalized_gap'],
    texttemplate='%{text}',
    textposition='outside',
    marker_color='seagreen'
))

# Layout settings
fig_winrate_gap_normalized.update_layout(
    title='Normalized Win Rate Gap by Country',
    xaxis_title='Country',
    yaxis_title='Normalized Win % Gap',
    template='plotly_white',
    yaxis=dict(range=[0, 1]),
    height=500
)

# Yearly average goals for home and away teams by country
country_goals = combined_df.groupby('Country').agg(
    avg_home_goals=('HomeGoals', 'mean'),
    avg_away_goals=('AwayGoals', 'mean')
).round(2)

import plotly.graph_objs as go

# Sort by avg_home_goals descending
country_goals_sorted = country_goals.sort_values(by='avg_home_goals', ascending=False)

# Plot grouped bar chart
goal_bar_fig = go.Figure()

goal_bar_fig.add_trace(go.Bar(
    x=country_goals_sorted.index,
    y=country_goals_sorted['avg_home_goals'],
    name='Avg Home Goals',
    marker_color='royalblue'
))

goal_bar_fig.add_trace(go.Bar(
    x=country_goals_sorted.index,
    y=country_goals_sorted['avg_away_goals'],
    name='Avg Away Goals',
    marker_color='darkorange'
))

goal_bar_fig.update_layout(
    barmode='group',
    title='Average Goals: Home vs Away by Country',
    xaxis_title='Country',
    yaxis_title='Goals per Match',
    yaxis=dict(range=[0, country_goals_sorted.values.max() + 0.2]),
    template='plotly_white'
)


import plotly.graph_objs as go

# Copy the original DataFrame with goal averages
goal_diff_sorted = country_goals.copy()

# Calculate goal difference (home - away)
goal_diff_sorted['goal_difference'] = goal_diff_sorted['avg_home_goals'] - goal_diff_sorted['avg_away_goals']

# Sort countries by goal difference in descending order
goal_diff_sorted = goal_diff_sorted.sort_values(by='goal_difference', ascending=False).round(2)

# Create bar chart for goal difference
goal_diff_fig = go.Figure()

goal_diff_fig.add_trace(go.Bar(
    x=goal_diff_sorted.index,                          # Country names
    y=goal_diff_sorted['goal_difference'],             # Goal difference values
    marker_color='crimson',                            # Bar color
    text=goal_diff_sorted['goal_difference'],          # Display value on bar
    textposition='outside',
    name='Goal Difference (Home - Away)'
))

# Update layout settings
goal_diff_fig.update_layout(
    title='Average Goals Gap by Country',
    xaxis_title='Country',
    yaxis_title='Goal Difference',
    template='plotly_white'
)

country_goals = combined_df.groupby('Country').agg(
    avg_home_goals=('HomeGoals', 'mean'),
    avg_away_goals=('AwayGoals', 'mean')
).round(2)


# Use country_goals, not country_stats
country_goals['goal_difference'] = country_goals['avg_home_goals'] - country_goals['avg_away_goals']

# Normalize the goal difference
country_goals['normalized_goal_diff'] = (
    (country_goals['goal_difference'] - country_goals['goal_difference'].min()) /
    (country_goals['goal_difference'].max() - country_goals['goal_difference'].min())
).round(2)

# Sort for plotting
goal_sorted = country_goals.sort_values(by='normalized_goal_diff', ascending=False)



import plotly.graph_objs as go

fig_normalized_goal_diff = go.Figure()

fig_normalized_goal_diff.add_trace(go.Bar(
    x=goal_sorted.index,
    y=goal_sorted['normalized_goal_diff'],
    text=goal_sorted['normalized_goal_diff'],
    texttemplate='%{text}',
    textposition='outside',
    marker_color='crimson'
))

fig_normalized_goal_diff.update_layout(
    title='Normalized Average Goals Gap by Country',
    xaxis_title='Country',
    yaxis_title='Normalized Goal Difference',
    template='plotly_white',
    yaxis=dict(range=[0, 1]),
    height=500
)

fig_normalized_goal_diff.show()


# Merge on 'Country' index
final_score_df = pd.merge(
    country_stats[['normalized_gap']],
    country_goals[['normalized_goal_diff']],
    left_index=True,
    right_index=True
)


# Calculate average score of both normalized metrics
final_score_df['overall_normalized_score'] = (
    (final_score_df['normalized_gap'] + final_score_df['normalized_goal_diff']) / 2
).round(2)

# Sort the final table
final_score_df = final_score_df.sort_values(by='overall_normalized_score', ascending=False)


import plotly.graph_objs as go

fig_final = go.Figure()

fig_final.add_trace(go.Bar(
    x=final_score_df.index,
    y=final_score_df['overall_normalized_score'],
    text=final_score_df['overall_normalized_score'],
    texttemplate='%{text}',
    textposition='outside',
    marker_color='darkorange'
))

fig_final.update_layout(
    title='Home Advantage Score by Country',
    xaxis_title='Country',
    yaxis_title='Overall Score (0–1)',
    template='plotly_white',
    yaxis=dict(range=[0, 1]),
    height=500
)

fig_final.show()


# Group by Country and Year, calculate win counts
country_year_stats = combined_df.groupby(['Country', 'Year']).agg(
    total_matches=('Result', 'count'),
    home_wins=('Result', lambda x: (x == 'H').sum()),
    away_wins=('Result', lambda x: (x == 'A').sum())
)

# Calculate win percentages
country_year_stats['home_win_pct'] = (country_year_stats['home_wins'] / country_year_stats['total_matches']) * 100
country_year_stats['away_win_pct'] = (country_year_stats['away_wins'] / country_year_stats['total_matches']) * 100

# Calculate win percentage gap
country_year_stats['win_gap'] = country_year_stats['home_win_pct'] - country_year_stats['away_win_pct']

# Reset index for plotting
country_year_stats = country_year_stats.reset_index()


import plotly.graph_objs as go

# Create figure
win_gap_trend_fig = go.Figure()

# Add a line per country
for country in country_year_stats['Country'].unique():
    subset = country_year_stats[country_year_stats['Country'] == country]

    win_gap_trend_fig.add_trace(go.Scatter(
        x=subset['Year'],
        y=subset['win_gap'],
        mode='lines',
        name=country
    ))

# Layout ayarları
win_gap_trend_fig.update_layout(
    title='Win Rate Gap Over the Years by Country',
    xaxis_title='Year',
    yaxis_title='Win % Gap',
    template='plotly_white',
    legend_title='Country',
    height=500
)

win_gap_trend_fig.show()


# Group by Country and Year, calculate average goals
country_year_goals = combined_df.groupby(['Country', 'Year']).agg(
    avg_home_goals=('HomeGoals', 'mean'),
    avg_away_goals=('AwayGoals', 'mean')
)

# Calculate goal difference (home - away)
country_year_goals['goal_gap'] = country_year_goals['avg_home_goals'] - country_year_goals['avg_away_goals']

# Reset index for plotting
country_year_goals = country_year_goals.reset_index()


import numpy as np

# Create empty list to collect results
trend_slopes = []

# Loop through each country and fit linear trend (1st degree polynomial)
for country in country_year_stats['Country'].unique():
    subset = country_year_stats[country_year_stats['Country'] == country]
    
    # Only calculate if country has enough years
    if len(subset) > 1:
        x = subset['Year']
        y = subset['win_gap']
        
        # Fit a 1st degree polynomial (linear)
        z = np.polyfit(x, y, 1)  # z[0] = slope
        trend_slopes.append({'Country': country, 'TrendSlope': round(z[0], 4)})

# Convert to DataFrame
import pandas as pd
trend_df = pd.DataFrame(trend_slopes)

# Sort by slope descending
trend_df = trend_df.sort_values(by='TrendSlope', ascending=False).reset_index(drop=True)


import plotly.graph_objs as go

win_gap_trend_slope_fig = go.Figure()

win_gap_trend_slope_fig.add_trace(go.Bar(
    x=trend_df['Country'],
    y=trend_df['TrendSlope'],
    text=trend_df['TrendSlope'],
    texttemplate='%{text:.4f}',
    textposition='outside',
    marker_color='teal'
))

win_gap_trend_slope_fig.update_layout(
    title='Trend Slope of Win Rate Gap Over the Years by Country',
    xaxis_title='Country',
    yaxis_title='Slope (Change per Year)',
    template='plotly_white',
    yaxis=dict(zeroline=True)
)

win_gap_trend_slope_fig.show()


# Invert the original trend slopes (to make more decline = higher value)
trend_df['inverted_slope'] = trend_df['TrendSlope'] * -1

# Apply min-max normalization on the inverted slopes
trend_df['normalized_decline'] = (
    (trend_df['inverted_slope'] - trend_df['inverted_slope'].min()) /
    (trend_df['inverted_slope'].max() - trend_df['inverted_slope'].min())
).round(2)

# Sort for clean visual
trend_sorted = trend_df.sort_values(by='normalized_decline', ascending=False)


import plotly.graph_objs as go

normalized_win_gap = go.Figure()

normalized_win_gap.add_trace(go.Bar(
    x=trend_sorted['Country'],
    y=trend_sorted['normalized_decline'],
    text=trend_sorted['normalized_decline'],
    texttemplate='%{text}',
    textposition='outside',
    marker_color='royalblue'
))

normalized_win_gap.update_layout(
    title='Normalized Trend Slope of Win Rate Gap Over the Years by Country',
    xaxis_title='Country',
    yaxis_title='Normalized Decline Score',
    template='plotly_white',
    yaxis=dict(range=[0, 1]),
    height=500
)

normalized_win_gap.show()


import plotly.graph_objs as go

goal_trend_fig = go.Figure()

# Add one line per country
for country in country_year_goals['Country'].unique():
    subset = country_year_goals[country_year_goals['Country'] == country]

    goal_trend_fig.add_trace(go.Scatter(
        x=subset['Year'],
        y=subset['goal_gap'],
        mode='lines',
        name=country
    ))

# Layout ayarları
goal_trend_fig.update_layout(
    title='Average Goal Gap Over the Years by Country',
    xaxis_title='Year',
    yaxis_title='Goal Difference',
    template='plotly_white',
    legend_title='Country',
    height=500
)

goal_trend_fig.show()


# Create empty list for goal slopes
goal_slopes = []

# Loop through each country
for country in country_year_goals['Country'].unique():
    subset = country_year_goals[country_year_goals['Country'] == country]

    if len(subset) > 1:
        x = subset['Year']
        y = subset['goal_gap']
        
        z = np.polyfit(x, y, 1)  # Linear fit
        goal_slopes.append({'Country': country, 'GoalSlope': round(z[0], 4)})

# Convert to DataFrame
goal_slope_df = pd.DataFrame(goal_slopes)

# Sort descending
goal_slope_df = goal_slope_df.sort_values(by='GoalSlope', ascending=False).reset_index(drop=True)


goal_trend_slope_fig = go.Figure()

goal_trend_slope_fig.add_trace(go.Bar(
    x=goal_slope_df['Country'],
    y=goal_slope_df['GoalSlope'],
    text=goal_slope_df['GoalSlope'],
    texttemplate='%{text:.4f}',
    textposition='outside',
    marker_color='indianred'
))

goal_trend_slope_fig.update_layout(
    title='Trend Slope of Average Goal Gap Over the Years by Country',
    xaxis_title='Country',
    yaxis_title='Slope (Goal Diff Change per Year)',
    template='plotly_white',
    yaxis=dict(zeroline=True)
)

goal_trend_slope_fig.show()


# Invert the goal difference slopes
goal_slope_df['inverted_goal_slope'] = goal_slope_df['GoalSlope'] * -1

# Normalize to 0–1
goal_slope_df['normalized_goal_decline'] = (
    (goal_slope_df['inverted_goal_slope'] - goal_slope_df['inverted_goal_slope'].min()) /
    (goal_slope_df['inverted_goal_slope'].max() - goal_slope_df['inverted_goal_slope'].min())
).round(2)

# Sort for plotting
goal_sorted = goal_slope_df.sort_values(by='normalized_goal_decline', ascending=False)


import plotly.graph_objs as go

goal_normalized = go.Figure()

goal_normalized.add_trace(go.Bar(
    x=goal_sorted['Country'],
    y=goal_sorted['normalized_goal_decline'],
    text=goal_sorted['normalized_goal_decline'],
    texttemplate='%{text}',
    textposition='outside',
    marker_color='crimson'
))

goal_normalized.update_layout(
    title='Normalized Trend Slope of Average Goal Gap Over the Years by Country',
    xaxis_title='Country',
    yaxis_title='Normalized Goal Decline Score',
    template='plotly_white',
    yaxis=dict(range=[0, 1]),
    height=500
)

goal_normalized.show()


# Merge two normalized tables on 'Country'
final_df = pd.merge(
    trend_sorted[['Country', 'normalized_decline']],            # From win_gap slope
    goal_sorted[['Country', 'normalized_goal_decline']],        # From goal_gap slope
    on='Country'
)

# Calculate average of both normalized scores
final_df['overall_decline_score'] = (
    (final_df['normalized_decline'] + final_df['normalized_goal_decline']) / 2
).round(2)

# Sort by final score
final_df = final_df.sort_values(by='overall_decline_score', ascending=False).reset_index(drop=True)


import plotly.graph_objs as go

final_score = go.Figure()

final_score.add_trace(go.Bar(
    x=final_df['Country'],
    y=final_df['overall_decline_score'],
    text=final_df['overall_decline_score'],
    texttemplate='%{text}',
    textposition='outside',
    marker_color='darkorange'
))

final_score.update_layout(
    title='Home Advantage Score Over the Years',
    xaxis_title='Country',
    yaxis_title='Overall Decline Score (0–1)',
    template='plotly_white',
    yaxis=dict(range=[0, 1]),
    height=500
)

final_score.show()


# Merge the two normalized scores tables on 'Country'
combined_scores = pd.merge(
    final_score_df.reset_index(),       # Contains overall_normalized_score, index = Country
    final_df[['Country', 'overall_decline_score']],  # Contains overall_decline_score
    on='Country'
)

# Calculate adjusted score by subtracting decline from advantage
combined_scores['adjusted_score'] = (
    combined_scores['overall_normalized_score'] - combined_scores['overall_decline_score']
).round(2)

# Sort by adjusted score descending
combined_scores = combined_scores.sort_values(by='adjusted_score', ascending=False).reset_index(drop=True)

# Show the result
print(combined_scores[['Country', 'overall_normalized_score', 'overall_decline_score', 'adjusted_score']])


import plotly.graph_objs as go

fig_final_summary = go.Figure()

fig_final_summary.add_trace(go.Bar(
    x=combined_scores['Country'],
    y=combined_scores['adjusted_score'],
    text=combined_scores['adjusted_score'],
    texttemplate='%{text}',
    textposition='outside',
    marker_color='mediumseagreen'
))

fig_final_summary.update_layout(
    title='Current State of Home Advantage',
    xaxis_title='Country',
    yaxis_title='Adjusted Score',
    template='plotly_white',
    height=500
)

fig_final_summary.show()


import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output


# Create the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Heimvorteil Analyse"

# Example graph names for each tab (can be customized)
graphs_tab1 = ['Grafik 1: Siegquote', 'Grafik 2: Tor-Durchschnitt']
graphs_tab2 = ['Grafik 3: Trend der Siegquote', 'Grafik 4: Trend des Tor-Durchschnitts']
graphs_tab3 = [
    'Grafik 5: Siegquote Heim–Auswärts nach Ländern',
    'Grafik 6: Siegquoten-Differenz Heim–Auswärts',
    'Grafik 7: Normalisierte Siegquoten-Differenz',
    'Grafik 8: Durchschnittliche Tore Heim–Auswärts nach Ländern',
    'Grafik 9: Tordifferenz Heim–Auswärts',
    'Grafik 10: Normalisierte Tordifferenz',
    'Grafik 11: Endergebnisse im Überblick'
]
graphs_tab4 = [
    'Grafik 12: Siegquoten-Differenz Heim–Auswärts im Zeitverlauf',
    'Grafik 13: Trend-Slope der Siegquoten-Differenz',
    'Grafik 14: Normalisierte Siegquoten-Differenz',
    'Grafik 15: Tordifferenz Heim–Auswärts im Zeitverlauf',
    'Grafik 16: Trend-Slope der Tordifferenz',
    'Grafik 17: Normalisierte Tordifferenz',
    'Grafik 18: Endergebnisse im Überblick 2'
]

app.layout = html.Div([
    # Title and general description
    html.H1("HEIMVORTEIL ANALYSE", style={'textAlign': 'center'}),
    html.H2("Inwiefern existiert ein Heimvorteil im europäischen Fußball – und wie hat sich dieser im Zeitverlauf entwickelt?", style={'textAlign': 'center'}),
    html.P("Diese Untersuchung basiert auf einem Datensatz mit den Ergebnissen von 216.883 Spielen " \
    "aus 22 Ligen in 11 Ländern im Zeitraum von 1993 bis 2023. Unter fünf thematischen Schwerpunkten " \
    "wurden insgesamt 19 verschiedene Grafiken erstellt und unterschiedliche Erkenntnisse gewonnen.",
           style={'textAlign': 'center'}),

    # Tab container
    dcc.Tabs([

        # TAB 1
        dcc.Tab(label='Heimvorteil', children=[
            html.H3("Analyse des Heimvorteils im europäischen Fußball"),
            html.P("In Grafik 1 werden die Siegquoten der Heim- und Auswärtsteams " \
            "über alle Spiele hinweg miteinander verglichen."),
            html.P("Grafik 2 zeigt den Vergleich der durchschnittlich erzielten Tore p" \
            "ro Spiel zwischen Heim- und Auswärtsteams."),
            html.P("Das Ergebnis: Mit einer Siegquote von 45 % und einem Tor-Durchschnitt " \
            "von 1,50 pro Spiel schneiden die Heimteams deutlich besser ab als die Auswärtsteams. " \
            "Dies bestätigt den allgemein bekannten Vorteil von Heimteams im Fußball."),

            
            dcc.Dropdown(
                id='tab1-dropdown',
                options=[
                    {'label': 'Grafik 1: Siegquote', 'value': 'Grafik 1: Siegquote'},
                    {'label': 'Grafik 2: Tor-Durchschnitt', 'value': 'Grafik 2: Tor-Durchschnitt'}
                ],
                value='Grafik 1: Siegquote'
            ),

            dcc.Graph(id='tab1-graph')
        ]),

        # TAB 2
        dcc.Tab(label='Die Entwicklung des Heimvorteils', children=[
            html.H3("Analyse der Entwicklung des Heimvorteils im europäischen Fußball im Zeitverlauf"),
            html.P("In Grafik 3 ist die Veränderung der Siegquoten von Heim- und Auswärtsteams " \
            "im Laufe der Jahre anhand von Trendlinien dargestellt."),
            html.P("In Grafik 4 ist die Veränderung der durchschnittlich erzielten Tore von Heim- " \
            "und Auswärtsteams im Laufe der Jahre anhand von Trendlinien dargestellt."),
            html.P("Sowohl bei den Siegquoten als auch bei den durchschnittlich erzielten Toren " \
            "ist über die Jahre ein Rückgang bei den Heimteams zu beobachten. Im Gegensatz dazu " \
            "zeigen die Auswärtsteams eine stetige Aufwärtsentwicklung. Trotz kleinerer " \
            "Schwankungen im Zeitverlauf gewinnen Heimteams insgesamt seltener und erzielen " \
            "weniger Tore, während Auswärtsteams immer erfolgreicher werden und häufiger treffen."),
            html.P("In diesem Abschnitt zeigt sich ein außergewöhnlicher Einfluss außerhalb des " \
            "eigentlichen Themas. In beiden Grafiken ist im Jahr 2020 eine deutliche negative " \
            "Abweichung vom Durchschnitt erkennbar. Diese Phase dauert etwa ein Jahr an, bevor " \
            "sich die Werte wieder erholen. Dieses Muster erinnert stark an die COVID-19-Pandemie. " \
            "In jener Zeit, in der Spiele ohne Zuschauer stattfanden und strenge Einschränkungen " \
            "galten, erreichte der Heimvorteil seinen niedrigsten Stand. Daraus lässt sich ableiten, " \
            "dass die Anwesenheit von Fans einen spürbaren Einfluss auf den Heimvorteil hat."),
            dcc.Dropdown(
                id='tab2-dropdown',
                options=[{'label': name, 'value': name} for name in graphs_tab2],
                value=graphs_tab2[0]
            ),
            dcc.Graph(id='tab2-graph')
        ]),

        # TAB 3
        dcc.Tab(label='Heimvorteil nach Ländern', children=[
            html.H3("Analyse des Heimvorteils im europäischen Fußball nach Ländern"),
            html.P("In Grafik 5 sind die Siegquoten von Heim- und Auswärtsteams nach " \
            "Ländern dargestellt. Die Länder sind dabei nach der Heim-Siegquote absteigend " \
            "sortiert. Diese Reihenfolge ist jedoch nicht vollständig aussagekräftig, " \
            "da die teils starken Schwankungen bei den Auswärtssiegquoten ins Auge fallen. " \
            "Für eine realistischere Bewertung ist nicht nur eine hohe Heim-Siegquote relevant, " \
            "sondern ebenso eine möglichst niedrige Auswärts-Siegquote."),
            html.P("Daher wird in Grafik 6 die Differenz zwischen Heim- und Auswärtssiegquote " \
            "berechnet und grafisch dargestellt. Dadurch ergibt sich eine aussagekräftigere " \
            "Rangfolge – so rückt beispielsweise Frankreich vom 8. auf den 2. Platz vor."),
            html.P("In Grafik 7 werden diese Differenzwerte auf einen Bereich von 0 bis 1 " \
            "normiert und in eine Punktzahl umgewandelt, um sie für weitere Auswertungen " \
            "verwenden zu können."),
            html.P("In Grafik 8 werden die durchschnittlichen Torzahlen von Heim- und " \
            "Auswärtsteams länderspezifisch dargestellt. Die Sortierung erfolgt erneut absteigend " \
            "nach dem Heimtor-Durchschnitt. Diese Reihenfolge ist jedoch wenig aussagekräftig " \
            "im Hinblick auf den Heimvorteil, da sie lediglich zeigt, in welchen Ländern generell " \
            "viele Tore erzielt werden – wie etwa in den Niederlanden."),
            html.P("Um die Leistung der Heimteams im Verhältnis zu den Auswärtsteams besser " \
            "zu bewerten, wird in Grafik 9 die Tordifferenz berechnet und als Grundlage für " \
            "die Rangfolge verwendet. Dadurch werden Länder hervorgehoben, in denen Heimteams " \
            "besonders erfolgreich sind. Beispielsweise steigt Griechenland von Platz 5 in " \
            "Grafik 8 auf Platz 1 in Grafik 9, während Schottland vom 5. auf den letzten Platz fällt."),
            html.P("In Grafik 10 werden diese Tordifferenzwerte auf einen Bereich von 0 bis 1 " \
            "normalisiert, um sie weiterverarbeiten zu können."),
            html.P("Grafik 11 kombiniert die Ergebnisse der Grafiken 7 und 10, um den Heimvorteil " \
            "länderspezifisch zu bewerten. Dabei wurden die normalisierten Werte der " \
            "Siegquoten-Differenz und der Tordifferenz jeweils zu 50 % gewichtet und zu einem " \
            "Gesamtwert zusammengeführt."),
            html.P("Das Ergebnis: Griechenland, das in beiden Einzelgrafiken den ersten Platz belegt, " \
            "führt das Ranking eindeutig an. England (Platz 10) und Schottland (Platz 11) behalten "
            "in beiden Tabellen ihre Positionen. Die übrigen Länder tauschen die Plätze untereinander " \
            "und werden im Gesamtranking auf Basis kleiner Punktunterschiede eingeordnet."),
            dcc.Dropdown(
                id='tab3-dropdown',
                options=[{'label': name, 'value': name} for name in graphs_tab3],
                value=graphs_tab3[0]
            ),
            dcc.Graph(id='tab3-graph')
        ]),

        # TAB 4
        dcc.Tab(label='Die Entwicklung des Heimvorteils nach Ländern', children=[
            html.H3("Analyse der Entwicklung des Heimvorteils im europäischen Fußball nach Ländern" \
            " im Zeitverlauf"),
            html.P("In Grafik 12 wird – basierend auf den Erkenntnissen aus der vorherigen Analyse" \
            " – direkt die Differenz der Siegquoten zwischen Heim- und Auswärtsteams betrachtet. " \
            "Für jedes Land wird diese Differenz berechnet und ihre Entwicklung über die Jahre " \
            "hinweg dargestellt. Aufgrund der Vielzahl an Linien wirkt die Darstellung zunächst " \
            "komplex, richtet sich jedoch an Leser, die einzelne Länder im Detail analysieren " \
            "möchten."),
            html.P("Um einen länderübergreifenden Vergleich zu ermöglichen, wird in Grafik 13 die" \
            " Trend-Slope dieser Differenz für jedes Land berechnet. Damit lässt sich erkennen, " \
            "wie sich der Heimvorteil im Zeitverlauf verändert hat. Alle Länder zeigen eine negative" \
            " Entwicklung – das bedeutet, dass der Heimvorteil überall abnimmt. Die geringste " \
            "Veränderung ist in Schottland zu beobachten, während Frankreich den stärksten " \
            "Rückgang verzeichnet."),
            html.P("In Grafik 14 werden die Trend-Slope-Werte auf einen Bereich von 0 bis 1 " \
            "normalisiert, um daraus standardisierte Punkte zu berechnen. Diese Punkte werden " \
            "später zur Bildung eines Gesamtscores verwendet."),
            html.P("In Grafik 15 wird diesmal die Differenz der durchschnittlichen Torzahlen " \
            "zwischen Heim- und Auswärtsteams berechnet und ihre Entwicklung über die Jahre " \
            "hinweg dargestellt. Wie bereits in Grafik 12 ergibt sich ein sehr unübersichtliches " \
            "Bild – diese Darstellung eignet sich daher vor allem zur Einzelbetrachtung der Länder."),
            html.P("Für einen besseren Vergleich wird in Grafik 16 die Trend-Slope der " \
            "Tordifferenz pro Land berechnet und in eine Rangfolge gebracht. Auch hier zeigt " \
            "sich in allen Ländern ein rückläufiger Trend. Den stärksten Rückgang verzeichnet " \
            "Griechenland, während die geringste Veränderung erneut in Schottland beobachtet wird."),
            html.P("Um diese Werte weiterverwenden zu können, werden sie in Grafik 17 auf " \
            "einen Bereich von 0 bis 1 normalisiert und in standardisierte Punktwerte umgerechnet."),
            html.P("In Grafik 18 werden die zuvor normalisierten Werte aus Grafik 14 und Grafik 17" \
            " – also die Punktwerte der Siegquoten- und Tordifferenz – jeweils zu 50 % gewichtet " \
            "und zu einem Gesamtscore kombiniert, um den Rückgang des Heimvorteils im Zeitverlauf " \
            "zu bewerten."),
            html.P("Das Ergebnis: Griechenland verzeichnet den stärksten Rückgang beim Heimvorteil " \
            "über die Jahre hinweg. Schottland, das in beiden Einzelwertungen den letzten Platz " \
            "belegt, zeigt erwartungsgemäß die geringste Veränderung. Die Entwicklung der übrigen " \
            "Länder kann im Detail der Grafik entnommen werden."),
            dcc.Dropdown(
                id='tab4-dropdown',
                options=[{'label': name, 'value': name} for name in graphs_tab4],
                value=graphs_tab4[0]
            ),
            dcc.Graph(id='tab4-graph')
        ]),

        # TAB 5
        dcc.Tab(label='Zusätzliche Analyse zum Heimvorteil nach Ländern', children=[
            html.H3("Verknüpfte Analyse des Heimvorteils nach Ländern und im Zeitverlauf"),
            html.P("In diesem Abschnitt wird eine Analyse durchgeführt, bei der die Punktwerte " \
            "aus Grafik 11 (Heimvorteil nach Ländern) mit den Veränderungswerten aus Grafik 17 "
            "(Rückgang des Heimvorteils über die Jahre) kombiniert werden. Grafik 11 und Grafik " \
            "17 sind unten nebeneinander dargestellt."),
            html.P("Für jedes Land wird der Punktwert aus Grafik 17 vom Punktwert aus Grafik 11 " \
            "subtrahiert, um einen neuen Gesamtwert zu erhalten. Basierend darauf erfolgt in " \
            "Grafik 19 eine weitere Rangfolge."),
            html.P("Ziel dieser letzten Darstellung ist es, aufzuzeigen, wie stark der " \
            "Heimvorteil in einem Land ursprünglich ausgeprägt war, wie stark er über die " \
            "Jahre zurückgegangen ist und wo das jeweilige Land aktuell im Vergleich steht."),
            html.P("Ein Beispiel dafür ist Griechenland und Schottland: Während Griechenland " \
            "ursprünglich den höchsten Heimvorteil hatte, verzeichnet es gleichzeitig den " \
            "stärksten Rückgang. Schottland hingegen weist den geringsten Heimvorteil auf, " \
            "zeigt aber nahezu keine Veränderung über die Jahre hinweg. Beide Länder landen " \
            "dadurch im kombinierten Ranking im Mittelfeld."),
            html.P("Betrachtet man Spanien, so liegt das Land mit einem Heimvorteilswert von " \
            "0,7 auf Platz 4. Gleichzeitig gehört es zu den Ländern mit dem geringsten Rückgang "
            "(Platz 2). Das bedeutet: Spanien gelingt es, seinen bestehenden Heimvorteil " \
            "weitgehend zu erhalten – was darauf hindeutet, dass das Land in den kommenden " \
            "Jahren das Potenzial hat, an die Spitze des Rankings aufzusteigen."),
            html.P("Betrachtet man die Niederlande, so liegt das Land beim Heimvorteil auf Platz 2 und bei dessen Rückgang nur an drittletzter Stelle. Das bedeutet, dass die Niederlande ihren Heimvorteil weitgehend bewahren konnten und im kombinierten Ranking den zweiten Platz belegen."),
            html.P("Auch Belgien, Deutschland und die Türkei gehören zu den Ländern, die ihren Heimvorteil zumindest teilweise erhalten haben. Sie belegen die Plätze 3, 4 und 5 in der Rangliste der Stabilität."),
            html.P("Portugal hingegen bildet das Schlusslicht dieser Analyse. Das Land liegt beim Heimvorteil nur auf Rang 9 und verzeichnet gleichzeitig den drittstärksten Rückgang. Dies deutet darauf hin, dass Portugal seinen Heimvorteil langfristig nicht bewahren kann und sich in Richtung des letzten Platzes bewegt."),
            html.P("Auch England, Frankreich und Italien gehören zu den Ländern, in denen der Heimvorteil besonders schnell abnimmt."),
           
            html.Div([
                html.Div(dcc.Graph(figure=fig_final), style={'width': '48%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(figure=final_score), style={'width': '48%', 'display': 'inline-block'})
            ]),

            html.Div([
                dcc.Graph(figure=fig_final_summary)
            ])
        ])
    ])
])

# Callback for Tab-1
@app.callback(
    Output('tab1-graph', 'figure'),
    Input('tab1-dropdown', 'value')
)
def update_tab1_graph(selected_value):
    if selected_value == 'Grafik 1: Siegquote':
        return match_fig
    elif selected_value == 'Grafik 2: Tor-Durchschnitt':
        return goal_fig

# Callback for Tab-2
@app.callback(
    Output('tab2-graph', 'figure'),
    Input('tab2-dropdown', 'value')
)
def update_tab2_graph(selected_value):
    if selected_value == 'Grafik 3: Trend der Siegquote':
        return fig_winrate_trend
    elif selected_value == 'Grafik 4: Trend des Tor-Durchschnitts':
        return fig_goals_avg
    else:
        return go.Figure()

# Callback for Tab-3
@app.callback(
    Output('tab3-graph', 'figure'),
    Input('tab3-dropdown', 'value')
)
def update_tab3_graph(selected_value):
    if selected_value == 'Grafik 5: Siegquote Heim–Auswärts nach Ländern':
        return win_bar_fig
    elif selected_value == 'Grafik 6: Siegquoten-Differenz Heim–Auswärts':
        return win_gap_fig
    elif selected_value == 'Grafik 7: Normalisierte Siegquoten-Differenz':
        return fig_winrate_gap_normalized
    elif selected_value == 'Grafik 8: Durchschnittliche Tore Heim–Auswärts nach Ländern':
        return goal_bar_fig
    elif selected_value == 'Grafik 9: Tordifferenz Heim–Auswärts':
        return goal_diff_fig
    elif selected_value == 'Grafik 10: Normalisierte Tordifferenz':
        return fig_normalized_goal_diff
    elif selected_value == 'Grafik 11: Endergebnisse im Überblick':
        return fig_final
    else:
        return go.Figure()

# Callback for Tab-4
@app.callback(
    Output('tab4-graph', 'figure'),
    Input('tab4-dropdown', 'value')
)
def update_tab4_graph(selected_value):
    if selected_value == 'Grafik 12: Siegquoten-Differenz Heim–Auswärts im Zeitverlauf':
        return win_gap_trend_fig
    elif selected_value == 'Grafik 13: Trend-Slope der Siegquoten-Differenz':
        return win_gap_trend_slope_fig
    elif selected_value == 'Grafik 14: Normalisierte Siegquoten-Differenz':
        return normalized_win_gap
    elif selected_value == 'Grafik 15: Tordifferenz Heim–Auswärts im Zeitverlauf':
        return goal_trend_fig
    elif selected_value == 'Grafik 16: Trend-Slope der Tordifferenz':
        return goal_trend_slope_fig
    elif selected_value == 'Grafik 17: Normalisierte Tordifferenz':
        return goal_normalized
    elif selected_value == 'Grafik 18: Endergebnisse im Überblick 2':
        return final_score
    else:
        return go.Figure()
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8051)

