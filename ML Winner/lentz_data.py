import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def combine_csvs(input_folder, output_file):
    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Iterate through each CSV file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Append the DataFrame to the combined_data DataFrame
            combined_data = combined_data.append(df, ignore_index=True)

    # Save the combined data to a new CSV file
    combined_data.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

def data_splits(csv_path, output_path, match_json_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Load the JSON file
    with open(match_json_path, 'r') as json_file:
        match_json = json.load(json_file)

    train = None
    valid = None
    test = None
    # Create output folders if they don't exist
    for major_num in range(1, 6):
        # Filter rows based on match IDs and save to a single CSV file for each folder
        qualifying_ids = match_json[f"major{major_num}"]['qualifying']
        event_ids = match_json[f"major{major_num}"]['event']
        filtered_df = df[df['matchGame.matchId'].isin(qualifying_ids + event_ids)]
        filtered_df = filtered_df.drop(columns = ["matchGame.matchId"])
        if major_num == 5:
            test = filtered_df
        elif major_num == 4:
            valid = filtered_df
        elif test is None:
            train = filtered_df
        else:
            test = pd.concat([test, filtered_df], ignore_index=True)

     # Filter rows based on match IDs and save to a single CSV file for each folder
    champs_ids = match_json["champs"]['event']
    print(champs_ids)
    filtered_df = df[df['matchGame.matchId'].isin(champs_ids)]
    filtered_df = filtered_df.drop(columns = ["matchGame.matchId"])
    filtered_df.to_csv(os.path.join(output_path, 'ablation.csv'), index=False)

    # Save to CSV files

    train.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    valid.to_csv(os.path.join(output_path, 'valid.csv'), index=False)
    test.to_csv(os.path.join(output_path, 'test.csv'), index=False)

    print("Data split and saved successfully.")

def time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

def clean_data(df, mode):
    # Drop columns with text or irrelevant information
    columns_to_drop = [
        "gameMap",'gameMode','id','programId','firstName','lastName','alias','socialNetworkHandles',
        'firstName', 'lastName', 'alias', 'headshot', 'socialNetworkHandles',
        'team_type', 'oppo_abbrev', 'abbrev', 'matchGame.id', 'matchGame.number', 'matchGame.gameMap.Version',
        'matchGame.gameMap.locale', 'matchGame.gameMap.uid', 'matchGame.gameMap.InProgress',
        'matchGame.gameMap.abbreviatedName', 'matchGame.gameMap.createdAt', 'matchGame.gameMap.createdBy',
        'matchGame.gameMap.description', 'matchGame.gameMap.desktopImage.src', 'matchGame.gameMap.displayName',
        'matchGame.gameMap.mapId', 'matchGame.gameMap.mobileImage.src', 'matchGame.gameMap.shareStatsImage.src',
        'matchGame.gameMap.tags', 'matchGame.gameMap.title', 'matchGame.gameMap.updatedAt', 'matchGame.gameMap.updatedBy',
        'matchGame.gameMap.publishDetails.environment', 'matchGame.gameMap.publishDetails.locale',
        'matchGame.gameMap.publishDetails.time', 'matchGame.gameMap.publishDetails.user',
        'matchGameResult.hostGameScore', 'matchGameResult.guestGameScore',
        'matchGameResult.winnerTeamId', 'matchGameResult.loserTeamId', 'homeTeamGamesWon', 'awayTeamGamesWon',
        'loserTeamId', 'matchDate', 'esportsTournamentId', 'matchGame.createdAt', 'matchGame.updatedAt',
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Convert columns with "Time" in their names to number of seconds
    time_columns = [col for col in df.columns if 'Time' in col]
    bad = ["deadSilenceTime", "percentTimeMoving", "totalTimeAlive", "hillTime", "contestedHillTime"]
    for col in time_columns:
        if col not in bad:
            df[col] = df[col].apply(time_to_seconds)
    
    # Drop columns with all 0's or NaNs
    df = df.dropna(axis=1, how='all')

    # Drop columns with sum equal to 0
    df = df.loc[:, df.sum() != 0]

    df["winner"] = np.where(df["team_id"] == df["winnerTeamId"], 1, 0)
    df = df.drop(columns = ["winnerTeamId", "team_id"])

    for col in df.columns:
         # Check if the values in the column are greater than 100
        if col != 'matchGame.matchId' and (df[col] > 100).any():
            # Divide values greater than 100 by 100
            df.loc[df[col] > 100, col] /= 100

    #remove all na data or data tied to win condition (dont know if this is neccesary)
    if mode == "Hardpoint":
        pass
    elif mode == "SnD":
        df = df.drop(columns = ["totalHeadshots"])
    else:
        df = df.drop(columns = ["totalHeadshots", "totalTiersCaptured"])

    # column_sums = df.sum()
    # print(column_sums)
    df = df.dropna()
    return df 

def split_csv_by_mode(csv_path, output_folder):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Create output folders if they don't exist
    for mode in ["CDL Control", "CDL SnD", "CDL Hardpoint"]:
        folder = f"{mode}"
        folder_path = os.path.join(output_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

        filtered_df = df[df['gameMode'] == mode]
        # filtered_df.drop(columns=['matchGame.matchId'])
        output_path = os.path.join(folder_path, f"{folder}.csv")
        filtered_df.to_csv(output_path, index=False)

    print("Splitting completed.")

# Example usage:
# combine_csvs("./data", "./lentz_final/full_data.csv")
split_csv_by_mode("./lentz_final/full_data.csv", "./lentz_final/")
mode = "Control"
csv_file_path = f"./lentz_final/CDL {mode}/CDL {mode}.csv"
output_folder_path = f"./lentz_final/CDL {mode}"
match_json = "./major_ids.json"

data = pd.read_csv(csv_file_path)
df = clean_data(data, f"{mode.lower()}")
df.to_csv(f"{output_folder_path}/{mode.lower()}_clean.csv", index = False)
data_splits(f"{output_folder_path}/{mode.lower()}_clean.csv", output_folder_path, match_json)




# Load your CSV data into a DataFrame (replace 'your_file.csv' with the actual file path)
# df = pd.read_csv("./lentz_final/CDL Hardpoint/hardpoint_clean.csv")

# Extract the features (X) and target variable if applicable
# Assuming 'target_column' is the column you want to predict and you want to exclude it from features
# X = df.drop(columns=['winner', "killDeathRatio"])


# Display the first few rows of the standardized and normalized DataFrames
# print("Standardized Data:")
# print(X_standardized.head())

# print("\nNormalized Data:")
# print(X_normalized.head())
