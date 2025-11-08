import pandas as pd
import re
import joblib
import requests
from bs4 import BeautifulSoup as bs
import argparse
from sklearn.metrics import accuracy_score

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
    'California': 'California'
}

# Standardize and clean names
def clean_team_name(name: str) -> str:
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
def predict_winner(team1_name, team2_name, week):
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

        week_games.append({
            'Team 1': team1,
            'Team 2': team2})
    print(week_games)

    weekly_results = []
    for matchup in week_games:
        winner, confidence = predict_winner(matchup['Team 1'], matchup['Team 2'], week)

        weekly_results.append({
            'Team 1': matchup['Team 1'],
            'Team 2': matchup['Team 2'], 
            'Winner': winner, 
            'Confidence': confidence})

    weekly_results = pd.DataFrame(weekly_results)
    weekly_results = weekly_results.sort_values(by='Confidence', ascending=False).reset_index(drop=True)

    print(f"\nPredictions for Week {week}:")
    print(weekly_results[['Team 1', 'Team 2', 'Winner', 'Confidence']])

    weekly_results.to_csv('Prediction/Weekly_Results_12.csv')
    return weekly_results

# Main Function
if __name__ == "__main__":
    # team1_name = input("Enter Team 1: ")
    # team2_name = input("Enter Team 2: ")
    week = input("Enter the Week: ")

    predict_weekly_slate(week)
