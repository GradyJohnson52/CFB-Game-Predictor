from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from datetime import datetime
import time
import random
import joblib
import re
import os

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

ALL_STATS = [
    "rush_def",
    "pass_def",
    "rush_off",
    "pass_off",
    "score_def",
    "score_off",
    "turn_def",
    "turn_off",
    "win_pct",
    "pred_rank",
    "sos"
]

STATS_DIR = "stats_cache"
MODEL_DIR = "trained_models"

def load_stats_from_cache(monday_date):
    path = os.path.join(STATS_DIR, f"{monday_date}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def write_stats_to_cache(monday_date, stats_dict):
    """
    Save each stat DataFrame in stats_dict to its own CSV file.
    stats_dict: {stat_name: DataFrame}
    """
    date_str = monday_date.strftime("%Y-%m-%d")
    folder = f"{STATS_DIR}/{date_str}"
    os.makedirs(folder, exist_ok=True)

    for stat_name, team_map in stats_dict.items():
        try:
            # Convert plain dict → DataFrame
            df = pd.DataFrame.from_dict(team_map, orient='index').dropna()
            df.index.name = "team"

            path = f"{folder}/{stat_name}.csv"
            df.to_csv(path)
            print(f"Wrote {stat_name} for {date_str} to cache")
        except Exception as e:
            print(f"Failed to write {stat_name} for {date_str}: {e}")

def normalize_raw_stat_dict(raw_dict):
    """
    input:
        { team: value }
    output:
        { team: { "L3": value, "Year": value } }
    """
    return {team: {"L3": val, "Year": val} for team, val in raw_dict.items()}

def load_stat_csv(path):
    df = pd.read_csv(path, index_col=0)
    return {team: {"L3": row["L3"], "Year": row["Year"]} for team, row in df.iterrows()}

def monday_before(date_str):
    """
    Convert any game date into the Monday before the game.
    Input: 'YYYY-MM-DD'
    Output: 'YYYY-MM-DD' (Monday)
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")

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
    name = name.replace('\xa0', ' ')
    name = re.sub(r'\s+', ' ', name)
    name = name.replace('–', '-')
    name = name.replace(' ', ' ')     
    return TEAM_MAPPING.get(team_name, team_name)

def flip_feature_row(row):
    """Invert signs for loser row to avoid identical duplicate pairs."""
    return [-v for v in row]

def scrape_stats(monday_date):
    """
    Scrape all stats for a given Monday date
    Returns a dictionary of stat_name -> {team: value}
    """
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
        "sos": "https://www.teamrankings.com/college-football/ranking/schedule-strength-by-other?date={}"
    }

    column_schema_L3 = {
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
        'sos': (1, 2)
    }

    column_schema_year = {
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

    for stat_name, url_template in urls.items():
        url = url_template.format(monday_date)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        time.sleep(random.uniform(0.1, 0.5))

        if not response.ok:
            print(f"Failed to fetch {url} (status {response.status_code})")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find("table", {'class':'tr-table'})
        if not table:
            print(f"No table for {stat_name} on {monday_date}")
            continue

        rows = table.find_all('tr')
        stat_data = {}
        for row in rows[1:]:
            cells = row.find_all('td')
            team_index, value_index_L3 = column_schema_L3.get(stat_name, (1, 2))
            team_index, value_index_year = column_schema_year.get(stat_name, (1, 2)) 
            try:
                team = standardize_team_name(clean_team_name(cells[team_index].text.strip()))
            except IndexError:
                continue

            # grab L3 and Year values using column schemas
            try:
                val_L3 = float(cells[value_index_L3].text.strip().replace(",", "").replace("%",""))
            except (ValueError, IndexError):
                val_L3 = None
            try:
                val_Year = float(cells[value_index_year].text.strip().replace(",", "").replace("%",""))
            except (ValueError, IndexError):
                val_Year = None

            stat_data[team] = {"L3": val_L3, "Year": val_Year}

        print(f"Scraped {stat_name} for {monday_date}")
        stats_for_date[stat_name] = stat_data

    return stats_for_date

def get_week_stats(monday_date):
    if isinstance(monday_date, str):
        monday_date = datetime.strptime(monday_date, "%Y-%m-%d")

    date_str = monday_date.strftime("%Y-%m-%d")

    stats = {}
    loaded_all = True

    for stat in ALL_STATS:
        path = f"{STATS_DIR}/{date_str}/{stat}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            stats[stat] = df.to_dict(orient='index')
        else:
            loaded_all = False

    if loaded_all:
        return stats

    # Otherwise – scrape
    scraped = scrape_stats(monday_date)
    for stat_name, team_map in scraped.items():
        if all(isinstance(v, (int, float, type(None))) for v in team_map.values()):
            scraped[stat_name] = normalize_raw_stat_dict(team_map)

    write_stats_to_cache(monday_date, scraped)
    return scraped

def build_feature_row(game_row, stats_df, feature_cols, lookback="Year", randomize=True):
    """
    Build a single feature vector for one game.
    lookback: "L3" or "Year" — which stats to use
    """
    t1 = game_row["team1"]
    t2 = game_row["team2"]

    row1 = []
    row2 = []
    try:
        row1 = [stats_df[stat][t1][lookback] for stat in feature_cols]
    except KeyError as e:
        raise KeyError(f"Missing stat or team in stats_df for team={t1}, stat={e}")
    try:
        row2 = [stats_df[stat][t2][lookback] for stat in feature_cols]
    except KeyError as e:
        raise KeyError(f"Missing stat or team in stats_df for team={t2}, stat={e}")


    diff = [a - b for a, b in zip(row1, row2)]
    return [float(x) for x in diff] #, (t1 == game_row["team1"])

def update_model(games_df, model, scaler, feature_cols, lookback="Year", stats_cache=None):
    """
    Update/train the model on games_df
    lookback: "L3" or "Year" — selects which stats to use
    Automatically fetches stats from the cache system
    """
    X = []
    Y = []

    for _, game in games_df.iterrows():
        week = game['week']
        date = game['date']
        winner = game['winner']
        loser = game['loser']

        monday = monday_before(date)
        game_stats = get_week_stats(monday)

        if random.random() > 0.5:
            team1, team2 = winner, loser
            label = 1  # team1 (winner) wins
        else:
            team1, team2 = loser, winner
            label = 0  # team1 (loser) loses

        try:
            # Build feature rows using selected lookback
            row = build_feature_row(
                {"team1": winner, "team2": loser, "date": date},
                game_stats,
                feature_cols,
                lookback=lookback
            )
            # row_lose = build_feature_row(
            #     {"team1": loser, "team2": winner, "date": date},
            #     game_stats,
            #     feature_cols,
            #     lookback=lookback
            # )
            #row_lose = flip_feature_row(row_lose)
        except Exception as e:
            print(f"Skipping {winner} vs {loser} on {date}: {e}")
            continue

        X.append(row)
        Y.append(label)  # winner

        # X.append(row_lose)
        # Y.append(0)  # loser

    # Save feature matrix for inspection
    df_features = pd.DataFrame(X, columns=[f"f{i}" for i in range(len(X[0]))])
    df_features['label'] = Y
    df_features.to_csv(f'training_features_{lookback}.csv', index=False)

    # Scale features
    X_scaled = scaler.fit_transform(X)

    # Fit the model
    model.fit(X_scaled, Y)
    print(f"Model training complete! Lookback={lookback}")
    return model, scaler

def test_meta_model(test_games, meta_nn, base_models, scalers, feature_cols, lookback="Year", csv_path="meta_test_results.csv"):
    """
    Test the stacked meta neural network model.

    Parameters:
    - test_games: dataframe of test games
    - meta_nn: trained Keras meta model
    - base_models: dict of base models
    - scalers: dict of corresponding scalers
    - feature_cols: feature names
    - lookback: "L3" or "Year"
    - csv_path: output path for results
    """

    X_meta_test = []
    Y_true = []
    predictions_detail = []

    for _, game in test_games.iterrows():
        date = game['date']
        winner = game['winner']
        loser = game['loser']
        week = game['week']

        monday = monday_before(date)
        game_stats = get_week_stats(monday)

        if random.random() > 0.5:
            team1, team2 = winner, loser
            team1_winner = True
        else:
            team1, team2 = loser, winner
            team1_winner = False

        # try:
        #     # Build L3 or Year features
        #     row = build_feature_row(
        #         {"team1": winner, "team2": loser, "date": date},
        #         game_stats,
        #         feature_cols,
        #         lookback=lookback
        #     )
        #     # row_lose = build_feature_row(
        #     #     {"team1": loser, "team2": winner, "date": date},
        #     #     game_stats,
        #     #     feature_cols,
        #     #     lookback=lookback
        #     # )
        #     #row_lose = flip_feature_row(row_lose)
        # except Exception as e:
        #     print(f"Skipping {winner} vs {loser}: {e}")
        #     continue

        base_preds = []
        skip_game = False

        for name, model in base_models.items():
            lookback = "L3" if "L3" in name else "Year"
            scaler = scalers[name]

            try:
                row = build_feature_row(
                    {"team1": team1, "team2": team2, "date": date},
                    game_stats,
                    feature_cols,
                    lookback=lookback
                )
            except Exception as e:
                print(f"Skipping base model {name}: {e}")
                skip_game = True
                break
            try:
                features_scaled = scaler.transform([row])

                if "REG" in name:
                    mov = model.predict(features_scaled)[0]
                    base_preds.append(np.tanh(mov / 30))
                else:
                    prob = model.predict_proba(features_scaled)[0][1]
                    base_preds.append(prob)
            except:
                print(f"Model {name} prediction failed for {team1} vs {team2}: {e}")
                skip_game = True
                break
        if skip_game:
            print(f"Skipping game, {team1} vs {team2} (got {len(base_preds)}) predictions)")
            continue

                #base_preds.append(week)

        X_meta_test.append(base_preds)
        Y_true.append(1 if team1_winner else 0)

        predictions_detail.append({
            'team1': team1,
            'team2': team2,
            'actual_winner': winner,
            'actual_winner_is_team1': team1_winner
        })
        


        # base_preds_lose = []

        # for name, model in base_models.items():
        #     scaler = scalers[name]

        #     features_scaled = scaler.transform([row_lose])

        #     if "REG" in name:
        #         mov = model.predict(features_scaled)[0]
        #         mov = np.tanh(mov/30)
        #         base_preds_lose.append(-mov)
        #     else:
        #         prob = model.predict_proba(features_scaled)[0][1]
        #         base_preds_lose.append(prob)

        # #base_preds_lose.append(week)

        # # lose case
        # X_meta_test.append(base_preds_lose)
        # Y_true.append(0)
    
    X_meta_test = np.array(X_meta_test)
    Y_true = np.array(Y_true)

    meta_probs = meta_nn.predict(X_meta_test)

    if len(meta_probs.shape) > 1:
        meta_probs_flat = meta_probs.flatten()
    else:
        meta_probs_flat = meta_probs

    y_pred = (meta_probs > 0.5).astype(int)

    assert len(y_pred) == len(Y_true), f"Prediction length mismatch: {len(y_pred)} vs {len(Y_true)}"
    assert len(meta_probs_flat) == len(Y_true), f"Probability length mismatch: {len(meta_probs_flat)} vs {len(Y_true)}"

    report = classification_report(Y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(csv_path, index=True)

    # pred_df = pd.DataFrame(predictions_detail)
    # pred_df['predicted_team1_wins'] = y_pred
    # pred_df['confidence'] = meta_probs.flatten()
    # pred_df['correct'] = (y_pred == Y_true)
    # pred_df.to_csv(csv_path.replace('.csv', '_detailed.csv'), index=False)
    
    print(f"Meta model test report saved to {csv_path}")
    print(f"Accuracy: {accuracy_score(Y_true, y_pred):.4f}")
    
    return df

def train_base_models(games_df, feature_cols):
    """
    Train 6 base models:
    RF_L3, RF_Year, XGB_L3, XGB_Year, REG_L3, REG_Year (MOV)
    Returns trained models dictionary and scalers
    """
    base_models = {}
    scalers = {}

    games_df["monday"] = games_df["date"].apply(monday_before)
    unique_mondays = games_df["monday"].unique()

    print(f"Precomputing stats for {len(unique_mondays)} Mondays")

    week_stats_cache = {}
    for mon in unique_mondays:
        week_stats_cache[mon] = get_week_stats(mon)

    # Model definitions
    model_defs = {
        "RF_L3": RandomForestClassifier(n_estimators=100),
        "RF_Year": RandomForestClassifier(n_estimators=100),
        "XGB_L3": XGBClassifier(max_depth=4, use_label_encoder=False, eval_metric='logloss'),
        "XGB_Year": XGBClassifier(max_depth=4, use_label_encoder=False, eval_metric='logloss')
        # "REG_L3": GradientBoostingRegressor(n_estimators=100),
        # "REG_Year": GradientBoostingRegressor(n_estimators=100)
    }

    for name, model in model_defs.items():
        lookback = "L3" if "L3" in name else "Year"
        scaler = StandardScaler()
        print(f"Training {name} (lookback={lookback})")
        if "REG" in name:
            # Train regressor on point differential
            X, Y = [], []
            for _, game in games_df.iterrows():
                monday = monday_before(game['date'])
                stats = get_week_stats(monday)

                if random.random() > 0.5:
                    team1, team2 = game['winner'], game['loser']
                    point_diff = game['point_diff']  # Positive (team1 won)
                else:
                    team1, team2 = game['loser'], game['winner']
                    point_diff = -game['point_diff'] 
                try:
                    row = build_feature_row(
                        {"team1": game['winner'], "team2": game['loser'], "date": game['date']},
                        stats, feature_cols, lookback=lookback
                    )
                    X.append(row)
                    Y.append(point_diff)
                except:
                    continue
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, Y)
        else:
            # Train classifier on win/loss
            model, scaler = update_model(games_df, model, scaler, feature_cols, lookback)
        base_models[name] = model
        scalers[name] = scaler
    return base_models, scalers

def build_meta_features(games_df, base_models, scalers, feature_cols):
    """
    Generate meta-features from base models:
    Probability predictions (RF/XGB) or normalized MOV (REG)
    Returns (X_meta, Y_meta)
    """
    X_meta = []
    Y_meta = []

    for _, game in games_df.iterrows():
        week = game['week']
        date = game['date']
        winner = game['winner']
        loser = game['loser']

        monday = monday_before(date)
        stats = get_week_stats(monday)

        if random.random() > 0.5:
            team1, team2 = winner, loser
            label = 1
        else:
            team1, team2 = loser, winner
            label = 0

        meta_row = []
        failed = False

        for name, model in base_models.items():
            lookback = "L3" if "L3" in name else "Year"
            scaler = scalers[name]
            try:
                features = build_feature_row(
                    {"team1": winner, "team2": loser, "date": date},
                    stats, feature_cols, lookback
                )
            except:
                continue

            features_scaled = scaler.transform([features])
            if "REG" in name:
                mov_est = model.predict(features_scaled)[0]
                # Normalize MOV to roughly [-1, 1] range w/ 30 point scale
                meta_row.append(np.tanh(mov_est / 30))
            else:
                prob = model.predict_proba(features_scaled)[0][1]
                meta_row.append(prob)
        if len(meta_row) == len(base_models):
            X_meta.append(meta_row)
            Y_meta.append(label)
            # add loser row
            # meta_row_l = []
            # for i, name in enumerate(base_models.keys()):
            #     val = X_meta[-1][i]
            #     if "REG" in name:
            #         meta_row_l.append(-val)
            #     else:
            #         meta_row_l.append(1 - val)
            # X_meta.append(meta_row_l)
            # Y_meta.append(0)
    return np.array(X_meta), np.array(Y_meta)

def train_meta_nn(X_meta, Y_meta, epochs=50, batch_size=16):
    """
    Train small feedforward NN on meta-features
    """
    input_dim = X_meta.shape[1]
    nn = Sequential([
        Dense(16, input_dim=input_dim, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn.fit(X_meta, Y_meta, epochs=epochs, batch_size=batch_size, verbose=1)
    return nn

def train_stacked_pipeline(games_df, feature_cols, epochs=50, batch_size=16):
    """
    Train the full stacked model:
      - 6 base models (RF/XGB/REG, L3 & Year)
      - Meta NN
    Returns:
      base_models, scalers, trained_meta_nn
    """
    # Train base models
    print("Training base models")
    base_models, scalers = train_base_models(games_df, feature_cols)

    # Build meta features
    print("Building meta features")
    X_meta, Y_meta = build_meta_features(games_df, base_models, scalers, feature_cols)

    # Train meta NN
    print("Training meta neural network")
    meta_nn = train_meta_nn(X_meta, Y_meta, epochs=epochs, batch_size=batch_size)

    # Save models
    joblib.dump(base_models, "base_models.pkl")
    joblib.dump(scalers, "scalers.pkl")
    meta_nn.save("meta_nn.h5")
    print("Training complete! Models saved as base_models.pkl, scalers.pkl, meta_nn.h5")

    return base_models, scalers, meta_nn

def main():
    """
    Main function to:
    1. Load game data
    2. Select features
    3. Train base models (L3 & Year)
    4. Build meta features
    5. Train meta neural network
    6. Save models and scalers for later use
    """
    games_df = pd.read_csv("GamesData.csv")
    games_df = games_df[games_df['year'] > 2013]
    feature_cols = [
        'rush_off', 'rush_def', 'pass_off', 'pass_def',
        'score_off', 'score_def', 'turn_off', 'turn_def',
        'win_pct', 'pred_rank', 'sos'
    ]
    training_games = games_df[games_df['year'] < 2023]
    training_games = training_games.sample(frac=1).reset_index(drop=True)

    cut = int(len(training_games) * 0.4)

    base_mod_train = training_games[:cut]
    meta_mod_train = training_games[cut:]
    test_games = games_df[games_df['year'] >= 2023]
    # add on: [games_df['week'] > 4][games_df['year'] != 2021]

    # train_batch = training_games[800:1000]
    # test_batch = test_games[300:400]


    print("Training base models")
    base_models, scalers = train_base_models(base_mod_train, feature_cols)

    print("Building meta features")
    X_meta, Y_meta = build_meta_features(meta_mod_train, base_models, scalers, feature_cols)

    print("Training meta neural network")
    meta_nn = train_meta_nn(X_meta, Y_meta, epochs=50, batch_size=16)

    test_meta_model(test_games, meta_nn, base_models, scalers, feature_cols)

    joblib.dump(base_models, "Ensemble_Files/base_models.pkl")
    joblib.dump(scalers, "Ensemble_Files/scalers.pkl")
    meta_nn.save("Ensemble_Files/meta_nn.h5")

    print("All models trained and saved:")
    print(" - Base models → base_models.pkl")
    print(" - Scalers → scalers.pkl")
    print(" - Meta neural net → meta_nn.h5")

# Run the pipeline
if __name__ == "__main__":
    main()