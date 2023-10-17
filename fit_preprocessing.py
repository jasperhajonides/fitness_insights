import fitparse
import pandas as pd
import os
from tqdm import tqdm


def load_fit_file(file_path):
    """
    Load a .fit file using fitparse library.

    Parameters:
        file_path (str): Path to the .fit file.

    Returns:
        fitfile: FitFile object representing the loaded .fit file.
    """
    fitfile = fitparse.FitFile(file_path)
    return fitfile


def extract_variables_from_fit_files(fit_files, directory, keyword, reverse=False):
    """
    Extract specified keyword variables from multiple .fit files.

    Parameters:
        fit_files (list): List of .fit file names.
        directory (str): Directory path where the .fit files are located.
        keyword (str): Keyword to extract from .fit files.
        reverse (bool, optional): Whether to loop through .fit files in reverse order. Default is False.

    Returns:
        pd.DataFrame: Dataframe containing the extracted variables for each .fit file.
    """
    df_all_files = pd.DataFrame()

    for file_nr, file in enumerate(fit_files):
        fit_data = load_fit_file(os.path.join(directory, file))

        # Determine the range to loop over
        range_indices = range(len(fit_data.messages) - 1, -1, -1) if reverse else range(len(fit_data.messages))

        for i in range_indices:
            if keyword in fit_data.messages[i].get_values().keys():
                value = fit_data.messages[i].get_values()[keyword]
                df_all_files.at[file_nr, keyword] = value

                # Adding the file name to the dataframe
                df_all_files.at[file_nr, 'file'] = file
                break

    return df_all_files


def process_multiple_keywords(fit_files, directory, keyword_list, reverse=False):
    """
    Process multiple keywords from .fit files and merge the results.

    Parameters:
        fit_files (list): List of .fit file names.
        directory (str): Directory path where the .fit files are located.
        keyword_list (list): List of keywords to extract from .fit files.
        reverse (bool, optional): Whether to loop through .fit files in reverse order. Default is False.

    Returns:
        pd.DataFrame: Dataframe containing the merged results of all extracted keywords.
    """
    df_list = []

    for keyword in tqdm(keyword_list):
        df = extract_variables_from_fit_files(fit_files, directory, keyword, reverse)
        df_list.append(df)

    # Merge all dataframes on 'file' column
    df_merged = pd.concat(df_list, axis=1)
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]  # Remove duplicated 'file' columns

    return df_merged
