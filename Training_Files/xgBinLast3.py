import requests
from bs4 import BeautifulSoup
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from datetime import datetime
import time
import random
import joblib
import re

# Dictionary to map inconsistent team names between sources
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

# Scrape game results from the table
def scrape_game_results():
    urls = {
        2015 : "https://www.sports-reference.com/cfb/years/2015-schedule.html",
        2016 : "https://www.sports-reference.com/cfb/years/2016-schedule.html",
        2017 : "https://www.sports-reference.com/cfb/years/2017-schedule.html",
        2018 : "https://www.sports-reference.com/cfb/years/2018-schedule.html",
        2019 : "https://www.sports-reference.com/cfb/years/2019-schedule.html",
        2020 : "https://www.sports-reference.com/cfb/years/2020-schedule.html",
        2021 : "https://www.sports-reference.com/cfb/years/2021-schedule.html",
        2022 : "https://www.sports-reference.com/cfb/years/2022-schedule.html",
        2023 : "https://www.sports-reference.com/cfb/years/2023-schedule.html",
        2024 : "https://www.sports-reference.com/cfb/years/2024-schedule.html",
    }

    games_data = []
    for year, url in urls.items():
        print(f"Scraping {url}...")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find("table", {'class':'sortable stats_table'})
        rows = table.find_all('tr')
        
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) > 0:
                try:
                    date = datetime.strptime(cells[1].text.strip(), "%b %d, %Y").strftime("%Y-%m-%d")
                    week = int(cells[0].text.strip())
                    winner = standardize_team_name(clean_team_name(cells[4].text.strip()))
                    loser = standardize_team_name(clean_team_name(cells[7].text.strip()))
                    winner_pts = int(cells[5].text.strip())
                    loser_pts = int(cells[8].text.strip())
                    point_diff = winner_pts - loser_pts
                except ValueError:
                    continue
                games_data.append({
                    'week' : week,
                    'date': date,
                    'winner': winner,
                    'loser': loser,
                    'point_diff': point_diff,
                    'year' : year
                })
    df = pd.DataFrame(games_data)
    df.to_csv('CSVs/GamesData.csv', index=False)
    return df
    

# Function to scrape stats for the day of a game
def scrape_stats(date):
    urls = {
        "rush_def": "https://www.teamrankings.com/college-football/stat/opponent-rushing-yards-per-game?date={}",
        "pass_def": "https://www.teamrankings.com/college-football/stat/opponent-passing-yards-per-game?date={}",
        "rush_off": "https://www.teamrankings.com/college-football/stat/rushing-yards-per-game?date={}",
        "pass_off": "https://www.teamrankings.com/college-football/stat/passing-yards-per-game?date={}",
        "score_def": "https://www.teamrankings.com/college-football/stat/opponent-points-per-game?date={}",
        "score_off": "https://www.teamrankings.com/college-football/stat/points-per-game?date={}",
        "win_pct": "https://www.teamrankings.com/ncf/trends/win_trends/?date={}",
        "turn_def": "https://www.teamrankings.com/college-football/stat/takeaways-per-game?date={}",
        "turn_off": "https://www.teamrankings.com/college-football/stat/giveaways-per-game?date={}",
        "pred_rank": "https://www.teamrankings.com/college-football/ranking/predictive-by-other?date={}",
        # "home_rat": "https://www.teamrankings.com/college-football/ranking/home-by-other?date={}",
        # "away_rat": "https://www.teamrankings.com/college-football/ranking/away-by-other?date={}",
        "sos": "https://www.teamrankings.com/college-football/ranking/schedule-strength-by-other?date={}"
    }

    column_schema_map = {
        'rush_def': (1, 3),
        'pass_def': (1, 3),
        'rush_off': (1, 3),
        'pass_off': (1, 3),
        'score_def': (1, 3),
        'score_off': (1, 3),
        'turn_def': (1, 3),
        'turn_off': (1, 3),
        'win_pct': (0, 2),
        'pred_rank': (1, 2),
        'home_rat': (1, 2),
        'away_rat': (1, 2),
        'sos': (1, 2)
    }

    stats_for_date = {}

    for stat_name, url in urls.items():
        url = url.format(date)
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        }
        response = requests.get(url, headers=headers)
        time.sleep(random.uniform(0.1, 0.75))
        if not response.ok:
            print(f"Failed to fetch {url} (status {response.status_code})")
            continue
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find("table", {'class':'tr-table'})

        if not table:
            print(f"No table found for stat '{stat_name}' at {url} (likely no data for {date})")
            continue
        rows = table.find_all('tr')
        
        stat_data = {}
        for row in rows[1:]:
            cells = row.find_all('td')
            team_index, value_index = column_schema_map.get(stat_name, (1, 2))
            try:
                team = standardize_team_name(clean_team_name(cells[team_index].text.strip()))
            except IndexError:
                continue
            try:
                value = float(cells[value_index].text.strip().replace(",", "").replace("%", ""))
            except ValueError:
                value = None  
            stat_data[team] = value
        print("scraped!")
        stats_for_date[stat_name] = stat_data
        
        
    return stats_for_date

# Function to update the ml model
def update_model(games_df, stats_dict, model, scaler):
    X = []
    Y = []

    for _, game in games_df.iterrows():
        week = game['week']
        date = game['date']
        winner = game['winner']
        loser = game['loser']
        point_diff = game['point_diff']
        
    
        print(stats_dict[date])
        game_stats = stats_dict[date]
        
        try: 
            stats_winner = {
                'rush_off': game_stats['rush_off'].get(winner),
                'rush_def': game_stats['rush_def'].get(winner),
                'pass_off': game_stats['pass_off'].get(winner),
                'pass_def': game_stats['pass_def'].get(winner),
                'score_off': game_stats['score_off'].get(winner),
                'score_def': game_stats['score_def'].get(winner),
                'turn_off': game_stats['turn_off'].get(winner),
                'turn_def': game_stats['turn_def'].get(winner),
                'pred_rank': game_stats['pred_rank'].get(winner),
                'sos': game_stats['sos'].get(winner),
                'win_pct': game_stats['win_pct'].get(winner)
            }

            stats_loser = {
                'rush_off': game_stats['rush_off'].get(loser),
                'rush_def': game_stats['rush_def'].get(loser),
                'pass_off': game_stats['pass_off'].get(loser),
                'pass_def': game_stats['pass_def'].get(loser),
                'score_off': game_stats['score_off'].get(loser),
                'score_def': game_stats['score_def'].get(loser),
                'turn_off': game_stats['turn_off'].get(loser),
                'turn_def': game_stats['turn_def'].get(loser),
                'pred_rank': game_stats['pred_rank'].get(loser),
                'sos': game_stats['sos'].get(loser),
                'win_pct': game_stats['win_pct'].get(loser)
            }

            if any(v is None for v in list(stats_winner.values()) + list(stats_loser.values())):
                print(f"Skipping {winner} vs {loser} on {date} due to missing stat(s).")
                continue

            # Winner 3 or 2
            features_win = [
                stats_winner['rush_off'] - stats_loser['rush_def'],
                stats_loser['rush_off'] - stats_winner['rush_def'],
                stats_winner['pass_off'] - stats_loser['pass_def'],
                stats_loser['pass_off'] - stats_winner['pass_def'],
                stats_winner['score_off'] - stats_loser['score_def'],
                stats_loser['score_off'] - stats_winner['score_def'],
                stats_winner['turn_off'] - stats_loser['turn_def'],
                stats_loser['turn_off'] - stats_winner['turn_def'],
                stats_winner['pred_rank'],
                stats_loser['pred_rank'],
                stats_winner['sos'],
                stats_loser['sos'],
                stats_winner['win_pct'],
                stats_loser['win_pct'],
                week
            ]
            X.append(features_win)
            Y.append(1)

            # Loser 1 or 0
            features_lose = [
                stats_loser['rush_off'] - stats_winner['rush_def'],
                stats_winner['rush_off'] - stats_loser['rush_def'],
                stats_loser['pass_off'] - stats_winner['pass_def'],
                stats_winner['pass_off'] - stats_loser['pass_def'],
                stats_loser['score_off'] - stats_winner['score_def'],
                stats_winner['score_off'] - stats_loser['score_def'],
                stats_loser['turn_off'] - stats_winner['turn_def'],
                stats_winner['turn_off'] - stats_loser['turn_def'],
                stats_loser['pred_rank'],
                stats_winner['pred_rank'],
                stats_loser['sos'],
                stats_winner['sos'],
                stats_loser['win_pct'],
                stats_winner['win_pct'],
                week
            ]
            X.append(features_lose)
            Y.append(0)

        except KeyError:
            print(f"Missing data for {date}")
            continue
        

    # team 1 is winner, 2 is loser
    df_features = pd.DataFrame(X, columns=[
        'rush_adv_team1', 'rush_adv_team2', 'pass_adv_team1', 'pass_adv_team2',
        'score_adv_team1', 'score_adv_team2', 'turnover_adv_team1', 'turnover_adv_team2',
        'pred_rank_team1', 'pred_rank_team2', 'sos_team1', 'sos_team2', 'WinPct_team1', 'WinPct_team2', 'week'
    ])
    df_features['label'] = Y
    df_features.to_csv('CSVs/training_features.csv', index=False)

    if not X:
        raise ValueError("No games had complete stat data. Cannot fit scaler on empty data.")
    X_scaled = scaler.fit_transform(X)


    # Fit the model with the new data
    model.fit(X_scaled, Y)
    print("model is running")
    return model

def test_model(test_games, stats_dict, model, scaler):
    X_test = []
    Y_test = []

    for _, game in test_games.iterrows():
        date = game['date']
        week = game['week']
        winner = game['winner']
        loser = game['loser']
        point_diff = game['point_diff']

        try:
            game_stats = stats_dict[date]
            stats_winner = {
                'rush_off': game_stats['rush_off'].get(winner),
                'rush_def': game_stats['rush_def'].get(winner),
                'pass_off': game_stats['pass_off'].get(winner),
                'pass_def': game_stats['pass_def'].get(winner),
                'score_off': game_stats['score_off'].get(winner),
                'score_def': game_stats['score_def'].get(winner),
                'turn_off': game_stats['turn_off'].get(winner),
                'turn_def': game_stats['turn_def'].get(winner),
                'pred_rank': game_stats['pred_rank'].get(winner),
                'sos': game_stats['sos'].get(winner),
                'win_pct': game_stats['win_pct'].get(winner)
            }
            stats_loser = {
                'rush_off': game_stats['rush_off'].get(loser),
                'rush_def': game_stats['rush_def'].get(loser),
                'pass_off': game_stats['pass_off'].get(loser),
                'pass_def': game_stats['pass_def'].get(loser),
                'score_off': game_stats['score_off'].get(loser),
                'score_def': game_stats['score_def'].get(loser),
                'turn_off': game_stats['turn_off'].get(loser),
                'turn_def': game_stats['turn_def'].get(loser),
                'pred_rank': game_stats['pred_rank'].get(loser),
                'sos': game_stats['sos'].get(loser),
                'win_pct': game_stats['win_pct'].get(loser)
            }

            if any(v is None for v in list(stats_winner.values()) + list(stats_loser.values())):
                continue

            features_win = [
                stats_winner['rush_off'] - stats_loser['rush_def'],
                stats_loser['rush_off'] - stats_winner['rush_def'],
                stats_winner['pass_off'] - stats_loser['pass_def'],
                stats_loser['pass_off'] - stats_winner['pass_def'],
                stats_winner['score_off'] - stats_loser['score_def'],
                stats_loser['score_off'] - stats_winner['score_def'],
                stats_winner['turn_off'] - stats_loser['turn_def'],
                stats_loser['turn_off'] - stats_winner['turn_def'],
                stats_winner['pred_rank'],
                stats_loser['pred_rank'],
                stats_winner['sos'],
                stats_loser['sos'],
                stats_winner['win_pct'],
                stats_loser['win_pct'],
                week
            ]
            X_test.append(features_win)
            Y_test.append(1)

            features_lose = [
                stats_loser['rush_off'] - stats_winner['rush_def'],
                stats_winner['rush_off'] - stats_loser['rush_def'],
                stats_loser['pass_off'] - stats_winner['pass_def'],
                stats_winner['pass_off'] - stats_loser['pass_def'],
                stats_loser['score_off'] - stats_winner['score_def'],
                stats_winner['score_off'] - stats_loser['score_def'],
                stats_loser['turn_off'] - stats_winner['turn_def'],
                stats_winner['turn_off'] - stats_loser['turn_def'],
                stats_loser['pred_rank'],
                stats_winner['pred_rank'],
                stats_loser['sos'],
                stats_winner['sos'],
                stats_loser['win_pct'],
                stats_winner['win_pct'],
                week
            ]
            X_test.append(features_lose)
            Y_test.append(0)

        except KeyError:
            continue

    if not X_test:
        print("No test data available for evaluation.")
        return

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # Exact classification report (3/2/1/0)
    report_dict = classification_report(Y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv("CSVs/res_XGBin_L3.csv")
    print("Report CSV")



# Main function to scrape games, update model, and iterate
def main():
    # Boosted ml model for multinomial classification
    model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
    )

    scaler = StandardScaler()

    # Scrape game results
    games_df = scrape_game_results()

    train_games = games_df[games_df['year'] < 2024].copy()
    test_games = games_df[games_df['year'] == 2024].copy()
    
    # Loop through each game and update the model
    stats_dict = {}
    unique_dates = sorted(games_df['date'].unique())
    for date in unique_dates:
        print(date)
        stats_dict[date] = scrape_stats(date)

    model = update_model(train_games, stats_dict, model, scaler)

    test_model(test_games, stats_dict, model, scaler)

    joblib.dump(model, 'XGBin_Last3.pkl')
    joblib.dump(scaler, 'scalerXGBin3.pkl')

    print("Model training complete!")

# Run the main function
main()
