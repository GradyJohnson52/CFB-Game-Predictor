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
team_mapping = {
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

def clean_team_name(raw_name):
    return re.sub(r'\([^)]*\)', '', raw_name).strip()

def standardize_team_name(team_name):
    return team_mapping.get(team_name, team_name)  

MAC_USER_AGENTS = [
    # Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",

    # Chrome
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.105 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.111 Safari/537.36",

    # Firefox
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13.3; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13.4; rv:123.0) Gecko/20100101 Firefox/123.0",

    # Edge
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.105 Safari/537.36 Edg/123.0.2420.65"
]

def headers():
    return {
        "User-Agent": random.choice(MAC_USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

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
        'rush_def': (1, 2),
        'pass_def': (1, 2),
        'rush_off': (1, 2),
        'pass_off': (1, 2),
        'score_def': (1, 2),
        'score_off': (1, 2),
        'turn_def': (1, 2),
        'turn_off': (1, 2),
        'win_pct': (0, 2),
        'pred_rank': (1, 2),
        'home_rat': (1, 2),
        'away_rat': (1, 2),
        'sos': (1, 2)
    }

    stats_for_date = {}

    for stat_name, url in urls.items():
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        }
        url = url.format(date)
        response = requests.get(url, headers= headers)
        time.sleep(random.uniform(0.1, 0.5))
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
            team_index, value_index = column_schema_map.get(stat_name, (1, 2))  # fallback default
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
            Y.append(3 if point_diff > 10 else 2)

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
            Y.append(1 if point_diff > 10 else 0)

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
            Y_test.append(3 if point_diff > 10 else 2)

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
            Y_test.append(1 if point_diff > 10 else 0)

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
    df_report.to_csv("CSVs/mult_XG_base.csv")
    print("Report CSV")

    # Binary result accuracy (win vs loss)
    y_true_result = [1 if y in (2, 3) else 0 for y in Y_test]
    y_pred_result = [1 if y in (2, 3) else 0 for y in y_pred]

    binary_acc = accuracy_score(y_true_result, y_pred_result)
    binary_df = pd.DataFrame({
        "Binary Accuracy": [binary_acc],
        "Correct Predictions": [(np.array(y_true_result) == np.array(y_pred_result)).sum()],
        "Total Predictions": [len(y_true_result)]
    })
    binary_df.to_csv("CSVs/res_XG_base.csv", index=False)
    print(f"Binary win/loss accuracy: {binary_acc:.3f}")


# Main function to scrape games, update model, and iterate
def main():
    # Boosted ml model for multinomial classification
    model = XGBClassifier(
    objective='multi:softprob',  
    num_class=4,
    eval_metric='mlogloss',
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

    joblib.dump(model, 'XGM_L3.pkl')
    joblib.dump(scaler, 'scalerXGML3.pkl')

    print("Model training complete!")

# Run the main function
main()
