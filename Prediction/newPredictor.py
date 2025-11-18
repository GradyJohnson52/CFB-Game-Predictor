import pandas as pd
import numpy as np
import re
import joblib
import requests
from bs4 import BeautifulSoup as bs
import argparse
from sklearn.metrics import accuracy_score

# Based on zero point favorite win prob to 1 point favorite win prob increase
SCALING_FACTOR = 0.013

# Values courtesy of Action Network: https://www.actionnetwork.com/ncaaf/college-football-home-field-advantage-every-team-2025
HOME_ADV_DICT = {
    "UAB": 4.0,
    "Georgia So": 3.8,
    "Washington St": 3.7,
    "SMU": 3.7,
    "LSU": 3.6,
    "Tulane": 3.5,
    "Kansas St": 3.5,
    "Temple": 3.5,
    "Ohio": 3.4,
    "Oklahoma": 3.4,
    "Alabama": 3.4,
    "Wyoming": 3.3,
    "Penn St": 3.3,
    "Texas Tech": 3.3,
    "Louisville": 3.3,
    "Utah": 3.3,
    "Notre Dame": 3.3,
    "Duke": 3.2,
    "Texas": 3.2,
    "J Madison": 3.1,
    "Rice": 3.1,
    "Buffalo": 3.1,
    "Utah St": 3.1,
    "Iowa St": 3.0,
    "Liberty": 3.0,
    "New Mexico St": 3.0,
    "Oregon St": 3.0,
    "Colorado": 2.9,
    "Syracuse": 2.9,
    "Missouri": 2.9,
    "West Virginia": 2.9,
    "Boise St": 2.9,
    "Virginia": 2.9,
    "Ohio St": 2.9,
    "Washington": 2.8,
    "California": 2.8,
    "Arizona St": 2.8,
    "USC": 2.8,
    "Minnesota": 2.8,
    "Louisiana Tech": 2.7,
    "Air Force": 2.7,
    "Iowa": 2.7,
    "Fresno St": 2.7,
    "W Kentucky": 2.7,
    "Toledo": 2.7,
    "Florida": 2.7,
    "Miami OH": 2.7,
    "Baylor": 2.6,
    "Virginia Tech": 2.6,
    "NC State": 2.6,
    "Navy": 2.6,
    "Auburn": 2.6,
    "South Carolina": 2.6,
    "Indiana": 2.6,
    "Oklahoma St": 2.5,
    "Michigan": 2.5,
    "Oregon": 2.5,
    "Tennessee": 2.5,
    "San Jose St": 2.5,
    "S Alabama": 2.5,
    "Mississippi St": 2.5,
    "UTSA": 2.5,
    "Hawai'i": 2.5,
    "UConn": 2.5,
    "Southern Miss": 2.4,
    "N Texas": 2.4,
    "UCF": 2.4,
    "Kansas": 2.4,
    "Arkansas St": 2.4,
    "Clemson": 2.4,
    "Memphis": 2.4,
    "W Michigan": 2.4,
    "Kentucky": 2.4,
    "Middle Tenn": 2.4,
    "UNLV": 2.3,
    "Ball St": 2.3,
    "Florida Intl": 2.3,
    "Pittsburgh": 2.3,
    "Arizona": 2.3,
    "Louisiana": 2.3,
    "Mississippi": 2.3,
    "Colorado St": 2.3,
    "BYU": 2.3,
    "S Florida": 2.2,
    "Nevada": 2.2,
    "Wake Forest": 2.2,
    "Texas A&M": 2.2,
    "Boston College": 2.2,
    "Northwestern": 2.1,
    "E Carolina": 2.1,
    "Maryland": 2.1,
    "Army": 2.1,
    "C Michigan": 2.1,
    "Coastal Car": 2.1,
    "Rutgers": 2.1,
    "Marshall": 2.0,
    "Florida St": 2.0,
    "Texas St": 2.0,
    "UTEP": 1.9,
    "E Michigan": 1.9,
    "App State": 1.9,
    "UL Monroe": 1.9,
    "TCU": 1.9,
    "Troy": 1.8,
    "North Carolina": 1.8,
    "Miami": 1.8,
    "Houston": 1.8,
    "Georgia Tech": 1.8,
    "Illinois": 1.7,
    "Arkansas": 1.7,
    "Wisconsin": 1.7,
    "New Mexico": 1.6,
    "UCLA": 1.6,
    "San Diego St": 1.6,
    "Cincinnati": 1.6,
    "Nebraska": 1.6,
    "Kent St": 1.5,
    "Old Dominion": 1.5,
    "Florida Atlantic": 1.5,
    "Charlotte": 1.4,
    "Georgia": 1.4,
    "Georgia St": 1.4,
    "UMass": 1.4,
    "Vanderbilt": 1.3,
    "Stanford": 1.3,
    "Bowling Green": 1.3,
    "Michigan St": 1.1,
    "Akron": 1.1,
    "Northern Illinois": 0.8,
    "Purdue": 0.8,
    "Tulsa": 0.7,
    "Jacksonville St": 2.5,
    "Sam Houston": 2.5,
    "Kennesaw St": 2.5,
    "Missouri St": 2.5,
    "Delaware": 2.5
}

TEAM_MAPPING = {
    'Navy': 'Navy',
    'Air Force': 'Air Force',
    'Utah': 'Utah',
    'Army': 'Army',
    'Jacksonville State': 'Jacksonville St',
    'Indiana': 'Indiana',
    'Missouri': 'Missouri',
    'James Madison': 'J Madison',
    'Oregon': 'Oregon',
    'Michigan': 'Michigan',
    'Florida State': 'Florida St',
    'South Florida': 'S Florida',
    'Georgia Tech': 'Georgia Tech',
    'Texas State': 'Texas St',
    'Arkansas': 'Arkansas',
    'Southern California': 'USC',
    'Texas A&M': 'Texas A&M',
    'Marshall': 'Marshall',
    'Rice': 'Rice',
    'Memphis': 'Memphis',
    'South Alabama': 'S Alabama',
    'Old Dominion': 'Old Dominion',
    'Notre Dame': 'Notre Dame',
    'Cincinnati': 'Cincinnati',
    'Mississippi': 'Mississippi',
    'East Carolina': 'E Carolina',
    'Ohio': 'Ohio',
    'North Texas': 'N Texas',
    'Arizona State': 'Arizona St',
    'Vanderbilt': 'Vanderbilt',
    'Virginia Tech': 'Virginia Tech',
    'Georgia': 'Georgia',
    'Louisiana Tech': 'Louisiana Tech',
    'Central Michigan': 'C Michigan',
    'Brigham Young': 'BYU',
    'San Diego State': 'San Diego St',
    'Texas Tech': 'Texas Tech',
    'Nevada-Las Vegas': 'UNLV',
    'Florida International': 'Florida Intl',
    'Tulane': 'Tulane',
    'Northwestern': 'Northwestern',
    'Liberty': 'Liberty',
    'Louisiana': 'Louisiana',
    'Central Florida': 'UCF',
    'Toledo': 'Toledo',
    'Iowa': 'Iowa',
    'Boise State': 'Boise St',
    'Coastal Carolina': 'Coastal Car',
    'Auburn': 'Auburn',
    'Tennessee': 'Tennessee',
    'Texas-San Antonio': 'UTSA',
    'Virginia': 'Virginia',
    'UCLA': 'UCLA',
    'Northern Illinois': 'N Illinois',
    'West Virginia': 'West Virginia',
    'Houston': 'Houston',
    'Iowa State': 'Iowa St',
    'Kennesaw State': 'Kennesaw St',
    'Southern Mississippi': 'Southern Miss',
    'North Carolina State': 'NC State',
    'Louisiana-Monroe': 'UL Monroe',
    'Western Michigan': 'W Michigan',
    'Arizona': 'Arizona',
    'Bowling Green': 'Bowling Green',
    'Miami (OH)': 'Miami OH',
    'Connecticut': 'UConn',
    'Miami (FL)': 'Miami',
    'New Mexico': 'New Mexico',
    'Louisville': 'Louisville',
    'Wyoming': 'Wyoming',
    'Washington': 'Washington',
    'Fresno State': 'Fresno St',
    'Kentucky': 'Kentucky',
    'Penn State': 'Penn St',
    'Kansas': 'Kansas',
    'Mississippi State': 'Mississippi St',
    'Temple': 'Temple',
    'Oklahoma': 'Oklahoma',
    'Kansas State': 'Kansas St',
    'Eastern Michigan': 'E Michigan',
    'Sam Houston': 'Sam Houston',
    'Ohio State': 'Ohio St',
    'Texas': 'Texas',
    'Utah State': 'Utah St',
    'Tulsa': 'Tulsa',
    'Purdue': 'Purdue',
    'Duke': 'Duke',
    'Baylor': 'Baylor',
    'Arkansas State': 'Arkansas St',
    'Colorado': 'Colorado',
    'Georgia Southern': 'Georgia So',
    'Colorado State': 'Colorado St',
    'Nebraska': 'Nebraska',
    'Oklahoma State': 'Oklahoma St',
    'Rutgers': 'Rutgers',
    'Clemson': 'Clemson',
    'Texas Christian': 'TCU',
    'Akron': 'Akron',
    'Wake Forest': 'Wake Forest',
    'Illinois': 'Illinois',
    'Pittsburgh': 'Pittsburgh',
    'Michigan State': 'Michigan St',
    'Alabama': 'Alabama',
    'Nevada': 'Nevada',
    'Buffalo': 'Buffalo',
    'Washington State': 'Washington St',
    'Appalachian State': 'App State',
    'Missouri State': 'Missouri St',
    'Western Kentucky': 'W Kentucky',
    'Florida': 'Florida',
    'North Carolina': 'North Carolina',
    'Ball State': 'Ball St',
    'San Jose State': 'San Jose St',
    'Wisconsin': 'Wisconsin',
    'Southern Methodist': 'SMU',
    'Georgia State': 'Georgia St',
    'Oregon State': 'Oregon St',
    'Alabama-Birmingham': 'UAB',
    'Syracuse': 'Syracuse',
    'Louisiana State': 'LSU',
    'Troy': 'Troy',
    "Hawaii": 'Hawai\'i',
    'South Carolina': 'South Carolina',
    'Texas-El Paso': 'UTEP',
    'Charlotte': 'Charlotte',
    'Minnesota': 'Minnesota',
    'Boston College': 'Boston College',
    'Middle Tennessee State': 'Middle Tenn',
    'Florida Atlantic': 'Florida Atlantic',
    'Kent State': 'Kent St',
    'Maryland': 'Maryland',
    'Stanford': 'Stanford',
    'Massachusetts': 'UMass',
    'New Mexico State': 'New Mexico St',
    'California': 'California',
    'Delaware': 'Delaware'
}

# Standardize and clean names
def clean_team_name(name):
    if not isinstance(name, str):
        return name
    name = re.sub(r"\([^)]*\d+[^)]*\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.strip()

def standardize_team_name(team_name):
    if not isinstance(team_name, str):
        return team_name

    # normalize whitespace and punctuation
    name = team_name.strip()
    name = name.replace('\xa0', ' ')  # replace non-breaking spaces
    name = re.sub(r'\s+', ' ', name)  # collapse double spaces
    name = name.replace('–', '-')     # normalize dash types
    name = name.replace(' ', ' ')     
    return TEAM_MAPPING.get(team_name, team_name)

# rating_df = pd.read_csv('CSVs/PR_Table.csv')

# mean_rating = rating_df['Rating'].mean()
# std_rating = rating_df['Rating'].std()

# rating_df['Home_adv'] = BASE_HOME_ADV + SCALING_FACTOR * ((mean_rating - rating_df['Rating'])/ std_rating)

# home_adv_dict = dict(zip(rating_df['Team'], rating_df['Home_adv'].round(2)))


# def home_field_adj(prob, is_home, adv_points=BASE_HOME_ADV):
#     offset = adv_points * SCALING_FACTOR
#     if is_home:
#         prob += offset
#     else:
#         prob-= offset
#     return prob

# Load the processed matchup data
matchup_df = pd.read_csv('CSVs/advanced_matchup_data.csv')

# Load the trained model
best_model = joblib.load('Model_pkls/RFB_Last3.pkl')
scaler = joblib.load('Model_pkls/scalerRFB3.pkl')



# Clean team names to remove stray whitespace and unify casing
matchup_df['week'] = matchup_df['week'].astype(int)
matchup_df['team1_std'] = matchup_df['team1'].map(lambda x: standardize_team_name(clean_team_name(str(x).strip())))
matchup_df['team2_std'] = matchup_df['team2'].map(lambda x: standardize_team_name(clean_team_name(str(x).strip())))

# Prediction function
def predict_winner(team1_name, team2_name, week, team1_home):
    team1_std = standardize_team_name(team1_name)
    team2_std = standardize_team_name(team2_name)
    week = int(week)

    matchup = matchup_df[
        ((matchup_df['team1_std'] == team1_std) & (matchup_df['team2_std'] == team2_std) & (matchup_df['week'] == week)) |
        ((matchup_df['team1_std'] == team2_std) & (matchup_df['team2_std'] == team1_std) & (matchup_df['week'] == week))
    ]

    if matchup.empty:
            print(f"No data for {team1_std} vs {team2_std} (Week {week})")
            print("Available team1 names:", matchup_df['team1_std'].unique()[:5])
            print("Available team2 names:", matchup_df['team2_std'].unique()[:5])
            return None, None

    row = matchup.iloc[0]

    feature_cols = [
        'rush_adv_team1', 'rush_adv_team2', 'pass_adv_team1', 'pass_adv_team2',
        'score_adv_team1', 'score_adv_team2', 'turnover_adv_team1', 'turnover_adv_team2',
        'pred_rank_team1', 'pred_rank_team2', 'sos_team1', 'sos_team2', 
        'WinPct_team1', 'WinPct_team2', 'week'
    ]

    X = scaler.transform([row[feature_cols].values])
    proba = best_model.predict_proba(X)[0]

    team1_win_prob = float(proba[1])
    team2_win_prob = float(proba[0])

    # Flip if matchup reversed in dataset
    if row['team1_std'] != team1_std:
        team1_win_prob, team2_win_prob = team2_win_prob, team1_win_prob

    if team1_home:
        team1_win_prob = team1_win_prob + (HOME_ADV_DICT[team1_std] * SCALING_FACTOR)
        team2_win_prob = 1- team1_win_prob
    else:
        team2_win_prob = team2_win_prob + (HOME_ADV_DICT[team2_std] * SCALING_FACTOR)
        team1_win_prob = 1- team2_win_prob


    winner = team1_std if team1_win_prob >= team2_win_prob else team2_std
    confidence = round(max(team1_win_prob, team2_win_prob), 3)

    print(f"\nPredicted Winner: {winner}")
    print(f"Confidence: {confidence}\n")

    return (winner, confidence)

# Weekly slate predictor
def predict_weekly_slate(week):
    """
    Scrape Sports Reference 2025 schedule, find all games for a given week,
    and predict each using the trained model.
    """
    try:
        week = int(week)
    except:
        print("Week must be an integer.")
        return None

    print(f"Scraping schedule for Week {week}...")

    schedule_url = "https://www.sports-reference.com/cfb/years/2025-schedule.html"
    response = requests.get(schedule_url)
    print(response)
    sched_soup = bs(response.content, 'html.parser')

    sched_table = sched_soup.find('table', {'class': 'sortable stats_table'})
    if sched_table is None:
        print("Could not find schedule table on Sports Reference.")
        return None

    games = sched_table.find_all('tr')
    week_games = []

    for game in games:
        cells = game.find_all('td')
        if not cells:
            continue

        # Extract info
        week_num = cells[0].text.strip()
        if week_num == '':
            continue

        try:
            week_num = int(week_num)
        except:
            continue

        if week_num != week:
            continue

        team1 = standardize_team_name(clean_team_name(cells[4].text.strip()))
        team2 = standardize_team_name(clean_team_name(cells[7].text.strip()))
        team1_home = False if cells[6].text.strip() == '@' else True

        week_games.append({
            'Team 1': team1,
            'Team 2': team2,
            'Team1 Home': team1_home})
    print(week_games)

    weekly_results = []
    for matchup in week_games:
        winner, confidence = predict_winner(matchup['Team 1'], matchup['Team 2'], week, matchup['Team1 Home'])

        weekly_results.append({
            'Team 1': matchup['Team 1'],
            'Team 2': matchup['Team 2'], 
            'Winner': winner, 
            'Confidence': confidence})

    weekly_results = pd.DataFrame(weekly_results)
    weekly_results = weekly_results.sort_values(by='Confidence', ascending=False).reset_index(drop=True)

    print(f"\nPredictions for Week {week}:")
    print(weekly_results[['Team 1', 'Team 2', 'Winner', 'Confidence']])

    weekly_results.to_csv('Prediction/Weekly_Results_12_homeadj.csv')
    return weekly_results

# Main Function
if __name__ == "__main__":
    # team1_name = input("Enter Team 1: ")
    # team2_name = input("Enter Team 2: ")
    week = input("Enter the Week: ")

    predict_weekly_slate(week)
