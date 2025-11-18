import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the CSV files
RD_Table = pd.read_csv('CSVs/RD_Table.csv', header=None)
PD_Table = pd.read_csv('CSVs/PD_Table.csv', header=None)
RO_Table = pd.read_csv('CSVs/RO_Table.csv', header=None)
PO_Table = pd.read_csv('CSVs/PO_Table.csv', header=None)
SD_Table = pd.read_csv('CSVs/SD_Table.csv', header=None)
SO_Table = pd.read_csv('CSVs/SO_Table.csv', header=None)
TD_Table = pd.read_csv('CSVs/TD_Table.csv', header=None)
TO_Table = pd.read_csv('CSVs/TO_Table.csv', header=None)
TR_Table = pd.read_csv('CSVs/TR_Table.csv', header=None)
PR_Table = pd.read_csv('CSVs/PR_Table.csv', header=None)
Home_Table = pd.read_csv('CSVs/Home_Table.csv', header=None)
Away_Table = pd.read_csv('CSVs/Away_Table.csv', header=None)
SOS_Table = pd.read_csv('CSVs/SOS_Table.csv', header=None)
WP_Table = pd.read_csv('CSVs/WP_Table.csv', header=None)


# Rename columns
RD_Table.columns = ['team', 'rush_def']
PD_Table.columns = ['team', 'pass_def']
RO_Table.columns = ['team', 'rush_off']
PO_Table.columns = ['team', 'pass_off']
SD_Table.columns = ['team', 'score_def']
SO_Table.columns = ['team', 'score_off']
TD_Table.columns = ['team', 'takeaways']
TO_Table.columns = ['team', 'giveaways']
TR_Table.columns = ['team', 'win_pct']
PR_Table.columns = ['team', 'P-rating']
# Home_Table.columns = ['team', 'H-rating']
# Away_Table.columns = ['team', 'A-rating']
SOS_Table.columns = ['team', 'SOS-rating']
WP_Table.columns = ['team', 'WinPct']



# Check for duplicate teams
for df_name, df in zip(
    ['RD_Table', 'PD_Table', 'RO_Table', 'PO_Table', 'SD_Table', 'SO_Table', 'TD_Table', 'TO_Table', 'TR_Table', 'PR_Table', 'SOS_Table', 'WP_Table'],
    [RD_Table, PD_Table, RO_Table, PO_Table, SD_Table, SO_Table, TD_Table, TO_Table, TR_Table, PR_Table, SOS_Table, WP_Table]):
    duplicates = df[df.duplicated(subset='team', keep=False)]
    if not duplicates.empty:
        print(f"Duplicate entries found in {df_name}:")
        print(duplicates)

# Merge all tables into one DataFrame based on team names
team_df = RD_Table.merge(PD_Table, on='team', how='left') \
    .merge(RO_Table, on='team', how='left') \
    .merge(PO_Table, on='team', how='left') \
    .merge(SD_Table, on='team', how='left') \
    .merge(SO_Table, on='team', how='left') \
    .merge(TD_Table, on='team', how='left') \
    .merge(TO_Table, on='team', how='left') \
    .merge(PR_Table, on='team', how='left') \
    .merge(SOS_Table, on='team', how='left') \
    .merge(WP_Table, on='team', how='left')
    # .merge(Home_Table, on='team', how='left') \
    # .merge(Away_Table, on='team', how='left') \

# Rename columns to match the features
team_df.columns = ['team', 'rush_def', 'rush_off', 'pass_off', 'pass_def', 'score_def', 'score_off', 
                    'takeaways', 'giveaways', 'P-rating', 'SOS-rating', 'WinPct']

print(team_df)
# Function to process and scale the data
def process_team_data(df):
    # Handle missing values for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Scale the features
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


# Function to create matchups between teams
def create_adv_match_data(df):
    features = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            for week in range(1, 23):
                team1 = df.iloc[i]
                team2 = df.iloc[j]

                feature_row = {
                    'team1': team1['team'],
                    'team2': team2['team'],

                    # Rushing advantage
                    'rush_adv_team1': team1['rush_off'] - team2['rush_def'],
                    'rush_adv_team2': team2['rush_off'] - team1['rush_def'],

                    # Passing advantage
                    'pass_adv_team1': team1['pass_off'] - team2['pass_def'],
                    'pass_adv_team2': team2['pass_off'] - team1['pass_def'],

                    # Scoring advantage
                    'score_adv_team1': team1['score_off'] - team2['score_def'],
                    'score_adv_team2': team2['score_off'] - team1['score_def'],

                    # Turnover advantage
                    'turnover_adv_team1': team1['takeaways'] - team2['giveaways'],
                    'turnover_adv_team2': team2['takeaways'] - team1['giveaways'],

                    # Predictive rating
                    'pred_rank_team1': team1['P-rating'],
                    'pred_rank_team2': team2['P-rating'],

                    # Strength of schedule
                    'sos_team1': team1['SOS-rating'],
                    'sos_team2': team2['SOS-rating'],

                    'WinPct_team1': team1['WinPct'],
                    'WinPct_team2': team2['WinPct'],

                    'week': week
                }

                features.append(feature_row)
    
    return pd.DataFrame(features)


processed_team_df = process_team_data(team_df)
# Check if all expected teams are present
teams_in_data = set(processed_team_df['team'])

# Generate matchups between teams
matchup_df = create_adv_match_data(processed_team_df)

# Save the matchup data to a CSv
matchup_df.to_csv('CSVs/advanced_matchup_data.csv', index=False)
print("Matchup data processed and saved.")
