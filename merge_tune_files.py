import os

import pandas as pd

from src import PROJECT_PATH


def merge_csv_files(directory: str, file_name: str) -> pd.DataFrame:
    """
    Reads all CSV files in a directory whose names contain 'supervised' and 'dependent',
    and merges them into a single pandas DataFrame.

    Parameters:
        directory (str): Path to the directory containing the CSV files.

    Returns:
        pd.DataFrame: A merged DataFrame containing all matching CSV data.
    """
    # List all files in the directory
    files = [f for f in os.listdir(directory) if file_name in f and f.endswith(".csv")]

    # Initialize an empty list to hold DataFrames
    dataframes = []

    for file in files:
        filepath = os.path.join(directory, file)
        try:
            # Read the CSV file and append it to the list
            df = pd.read_csv(filepath)
            new_columns = df.iloc[:, 0].values.tolist()
            df = df.iloc[:, 1:]
            df = pd.DataFrame([df.iloc[:, 0].values.tolist()], columns=new_columns)
            dataframes.append(df)
            print(f"Loaded: {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Merge all DataFrames into one
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        print(f"Merged {len(dataframes)} files.")
    else:
        merged_df = pd.DataFrame()
        print("No files matched the criteria.")

    return merged_df


file_names = ["BALE-aomic-subject-dependent_best_hyperparameters_bale",
              "BALE-aomic-subject-dependent_best_hyperparameters_supervised",
              "BALE-aomic-subject-independent_best_hyperparameters_bale",
              "BALE-aomic-subject-independent_best_hyperparameters_supervised"]
dir = os.path.join(PROJECT_PATH, "results2")

for file_name in file_names:
    merged_data = merge_csv_files(dir, file_name)
    merged_data = merged_data.groupby(
        ["batch_size", "projection_dim", "mlp_h_dim", "epochs"]).mean().reset_index()
    best_trial = merged_data.loc[merged_data["accuracy"].idxmax()]
    best_trial.to_csv(os.path.join(PROJECT_PATH, f"{file_name}.csv"))
    print(f"Best hyperparameters: {best_trial}")
