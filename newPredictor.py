import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup as bs
import argparse
from sklearn.metrics import accuracy_score

TEAM_MAPPING = {
    'Florida Intl': 'FIU',
    'Ohio St': 'Ohio State',
    'Iowa St': 'Iowa State',
    'Notre Dame': 'Notre Dame',
    'N Illinois': 'Northern Illinois',
    'Washington': 'Washington',
    'Texas': 'Texas',
    'LA Tech': 'Louisiana Tech',
    'Sam Hous St': 'Sam Houston State',
    'UL Monroe': 'Louisiana-Monroe',
    'Minnesota': 'Minnesota',
    'Indiana': 'Indiana',
    'Bowling Grn': 'Bowling Green',
    'Wisconsin': 'Wisconsin',
    'W Kentucky': 'Western Kentucky',
    'Tulane': 'Tulane',
    'Kentucky': 'Kentucky',
    'Rice': 'Rice',
    'Alabama': 'Alabama',
    'Oregon': 'Oregon',
    'BYU': 'Brigham Young',
    'Houston': 'Houston',
    'U Mass': 'Massachusetts',
    'UAB': 'Alabama-Birmingham',
    'Army': 'Army',
    'Temple': 'Temple',
    'Tennessee': 'Tennessee',
    'Penn St': 'Penn State',
    'Colorado': 'Colorado',
    'Miami (OH)': 'Miami (OH)',
    'Missouri': 'Missouri',
    'TX Christian': 'Texas Christian',
    'Louisiana': 'Louisiana',
    'Liberty': 'Liberty',
    'Air Force': 'Air Force',
    'Florida St': 'Florida State',
    'Nevada': 'Nevada',
    'Marshall': 'Marshall',
    'S Carolina': 'South Carolina',
    'Central Mich': 'Central Michigan',
    'Fresno St': 'Fresno State',
    'Georgia St': 'Georgia State',
    'Oklahoma': 'Oklahoma',
    'Iowa': 'Iowa',
    'James Mad': 'James Madison',
    'Toledo': 'Toledo',
    'Utah': 'Utah',
    'Charlotte': 'Charlotte',
    'Oregon St': 'Oregon State',
    'Texas St': 'Texas State',
    'Navy': 'Navy',
    'Auburn': 'Auburn',
    'Georgia': 'Georgia',
    'Michigan St': 'Michigan State',
    'Troy': 'Troy',
    'VA Tech': 'Virginia Tech',
    'Jksnville St': 'Jacksonville State',
    'Michigan': 'Michigan',
    'Ohio': 'Ohio',
    'Connecticut': 'Connecticut',
    'S Mississippi': 'Southern Mississippi',
    'Miami': 'Miami',
    'Cincinnati': 'Cincinnati',
    'Nebraska': 'Nebraska',
    'Duke': 'Duke',
    'Clemson': 'Clemson',
    'San Diego St': 'San Diego State',
    'App State': 'Appalachian State',
    'Hawaii': 'Hawaii',
    'Arizona St': 'Arizona State',
    'California': 'California',
    'Wyoming': 'Wyoming',
    'UCF': 'Central Florida',
    'Syracuse': 'Syracuse',
    'LSU': 'LSU',
    'E Michigan': 'Eastern Michigan',
    'Illinois': 'Illinois',
    'GA Tech': 'Georgia Tech',
    'W Michigan': 'Western Michigan',
    'Coastal Car': 'Coastal Carolina',
    'TX El Paso': 'Texas-El Paso (UTEP)',
    'NC State': 'North Carolina State',
    'S Methodist': 'Southern Methodist',
    'Florida': 'Florida',
    'Kansas St': 'Kansas State',
    'USC': 'Southern California',
    'UNLV': 'Nevada-Las Vegas',
    'San Jose St': 'San Jose State',
    'Northwestern': 'Northwestern',
    'N Carolina': 'North Carolina',
    'Mississippi': 'Mississippi',
    'Old Dominion': 'Old Dominion',
    'Colorado St': 'Colorado State',
    'Akron': 'Akron',
    'Rutgers': 'Rutgers',
    'Texas A&M': 'Texas A&M',
    'Fla Atlantic': 'Florida Atlantic',
    'UCLA': 'UCLA',
    'Miss State': 'Mississippi State',
    'Baylor': 'Baylor',
    'Kennesaw St': 'Kennesaw State',
    'N Mex State': 'New Mexico State',
    'E Carolina': 'East Carolina',
    'Maryland': 'Maryland',
    'Kansas': 'Kansas',
    'Louisville': 'Louisville',
    'Boise St': 'Boise State',
    'Middle Tenn': 'Middle Tennessee State',
    'Buffalo': 'Buffalo',
    'Arizona': 'Arizona',
    'Vanderbilt': 'Vanderbilt',
    'Arkansas St': 'Arkansas State',
    'S Alabama': 'South Alabama',
    'Kent St': 'Kent State',
    'Arkansas': 'Arkansas',
    'Utah St': 'Utah State',
    'Boston Col': 'Boston College',
    'GA Southern': 'Georgia Southern',
    'Pittsburgh': 'Pittsburgh',
    'W Virginia': 'West Virginia',
    'Memphis': 'Memphis',
    'Purdue': 'Purdue',
    'North Texas': 'North Texas',
    'Wash State': 'Washington State',
    'Virginia': 'Virginia',
    'UTSA': 'Texas-San Antonio (UTSA)',
    'Ball St': 'Ball State',
    'New Mexico': 'New Mexico',
    'Stanford': 'Stanford',
    'Oklahoma St': 'Oklahoma State',
    'Wake Forest': 'Wake Forest',
    'Texas Tech': 'Texas Tech',
    'S Florida': 'South Florida',
    'Tulsa': 'Tulsa'
}

def standardize_team_name(team_name):
    return TEAM_MAPPING.get(team_name, team_name)

# Load the processed matchup data
matchup_df = pd.read_csv('CSVs/advanced_matchup_data.csv')

# Load the trained model
best_model = joblib.load('RFB_Last3.pkl')
scaler = joblib.load('scalerRFB3.pkl')

# Prediction function
def predict_winner(team1_name, team2_name, week):

    team1_std = standardize_team_name(team1_name)
    team2_std = standardize_team_name(team2_name)

    if 'team1_std' not in matchup_df.columns or 'team2_std' not in matchup_df.columns:
        matchup_df['team1_std'] = matchup_df['team1'].map(standardize_team_name)
        matchup_df['team2_std'] = matchup_df['team2'].map(standardize_team_name)

    # Lookup matchup
    matchup = matchup_df[
        ((matchup_df['team1_std'] == team1_std) & (matchup_df['team2_std'] == team2_std) & (matchup_df['week'] == week)) |
        ((matchup_df['team1_std'] == team2_std) & (matchup_df['team2_std'] == team1_std) & (matchup_df['week'] == week))
    ]

    if matchup.empty:
        return {"error": f"No matchup found for {team1_name} vs {team2_name}, week {week}"}

    row = matchup.iloc[0]

    # Decide ordering
    if row['team1_std'] == team1_std:
        team_a, team_b = team1_std, team2_std
    else:
        team_a, team_b = team2_std, team1_std

    # Features
    feature_cols = [
        'rush_adv_team1', 'rush_adv_team2', 'pass_adv_team1', 'pass_adv_team2',
        'score_adv_team1', 'score_adv_team2', 'turnover_adv_team1', 'turnover_adv_team2',
        'pred_rank_team1', 'pred_rank_team2', 'sos_team1', 'sos_team2', 
        'WinPct_team1', 'WinPct_team2', 'week'
    ]
    X = scaler_rf.transform([row[feature_cols].values])

    # Predict
    prediction = model_rf.predict(X)[0]
    proba = model_rf.predict_proba(X)[0]

    team1_win_prob = float(proba[1])
    team2_win_prob = float(proba[0])

    # Flip if mismatch
    if team_a != team1_std:
        team1_win_prob, team2_win_prob = team2_win_prob, team1_win_prob
        prediction = 1 - prediction

    winner = team1_std if team1_win_prob >= team2_win_prob else team2_std
    confidence = round(max(team1_win_prob, team2_win_prob), 3)
    print(f"winner: {winner}, Conf: {confidence}")

    return {"winner": winner, "confidence": confidence}

# def pred_weekly_slate(week):
#     schedule_url = "https://www.sports-reference.com/cfb/years/2025-schedule.html"
#     response = requests.get(schedule_url)

#     sched_soup = bs(response.content, 'html.parser')

#     sched = sched_soup.find('table', {'class': 'sortable stats_table'})
#     games = sched.find_all('tr')
#     for game in games:
#         game_cells = game.find_all('td')




# Predict
team1_name = input("Enter Team 1:")
team2_name = input("Enter Team 2:")
week = input("Enter the Week:")
predict_winner(team1_name, team2_name, week)
