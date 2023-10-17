import numpy as np
import pandas as pd


def set_swimlap_indices(df):
    """Function adds counts for the lengths in a lap and the laps.

    Parameters:
        df (pandas.DataFrame): DataFrame containing swim data with 'event' and 'cum_lap_time' columns.

    Returns:
        pandas.DataFrame: DataFrame with added 'lap_nr', 'length_nr', and 'cum_lap_time' columns.
    """

    df = df.loc[df.event.isin(['length', 'lap'])].copy()

    df.loc[:, 'lap_nr'] = None
    df.loc[:, 'length_nr'] = None

    lap_count = 0
    length_count = 0
    # cum_lap_time = 0

    for index, row in df.iterrows():
        df.at[index, 'lap_nr'] = lap_count

        if row['event'] == 'lap':
            lap_count += 1
            length_count = 0
            # cum_lap_time = 0
        elif row['event'] == 'length':
            length_count += 1


            df.at[index, 'lap_nr'] = lap_count
            df.at[index, 'length_nr'] = length_count


    # narrow down the dataframe
    swimming_vars = ['sub_sport', 'total_distance', 'total_elapsed_time', 'total_cycles', 'avg_stroke_count',
                     'total_calories',
                     'avg_speed', 'num_laps', 'num_active_lengths', 'avg_heart_rate',
                     'swim_stroke', 'num_lengths', 'first_length_index', 'avg_swimming_cadence', 'total_strokes',
                     'length_type', 'elapsed_time',
                     'lap_nr', 'length_nr']
    df = df.iloc[:-1, :]  # drop last column because its the summed total
    df = df[swimming_vars]

    # add column for cumulative time
    df = add_cumulative_time(df)

    # get max times in duration and end times
    get_start_time_df = df.groupby('lap_nr')[['cumsum', 'elapsed_time']].max().dropna(
        subset='cumsum').reset_index()

    # calculate the start time of the interval
    get_start_time_df['start_time'] = get_start_time_df['elapsed_time'] - get_start_time_df[
        'cumsum']
    get_start_time_df.rename(columns={'cumsum': 'time_interval'}, inplace=True)  # rename

    # merge in the new variables
    df_laps = pd.merge(df, get_start_time_df[['start_time', 'lap_nr', 'time_interval']], on='lap_nr', how='left')

    # set format
    df_laps.lap_nr = df_laps.lap_nr.apply(int)

    # calculate SWOLF
    df_laps.loc[:,'SWOLF'] = df_laps['total_elapsed_time'] + df_laps['total_strokes']

    return df_laps


def add_cumulative_time(df):
    """Function adds cumulative time for lengths with 'length_type' equal to 'active'.

    Parameters:
        df (pandas.DataFrame): DataFrame containing swim data with 'length_type' and 'total_elapsed_time' columns.

    Returns:
        pandas.DataFrame: DataFrame with added 'cumsum' column.
    """
    save = 0
    for i, row in df.iterrows():
        if row['length_type'] != 'active':
            save = 0
        else:
            save += row['total_elapsed_time']
            df.at[i, 'cumsum'] = save

    return df


def obtain_hr_epochs(df, df_laps, distances=[50, 100, 200, 400]):
    """Function obtains heart rate epochs based on lap data and heart rate data.

    Parameters:
        df (pandas.DataFrame): DataFrame containing heart rate data with 'elapsed_time' and 'heart_rate' columns.
        df_laps (pandas.DataFrame): DataFrame containing lap data with 'start_time', 'time_interval', 'total_distance', and 'swim_stroke' columns.
        distances (list): List of distances of interest.

    Returns:
        dict: A dictionary containing dataframes for each distance with heart rate, time, pace, and rest data.
    """
    data = df.dropna(subset='heart_rate')[['elapsed_time', 'heart_rate']]

    time_intervals = df_laps.loc[df_laps['swim_stroke'] != 'drill', ['start_time', 'time_interval', 'total_distance']].drop_duplicates().dropna()
    time_intervals['rest'] = time_intervals['start_time'].diff() - time_intervals['time_interval'].shift()

    # Create a dictionary to hold the dataframes for each distance
    distance_data = {distance: {'heart_rate': [], 'time': [], 'pace': [], 'rest': []} for distance in distances}

    # Iterate over the rows in the lap data
    first_iteration = True
    for i, row in time_intervals.iterrows():
        # Check if the total distance for the row is in the list of distances
        if int(row['total_distance']) in distances:
            # Get the start and end times for the interval
            start_time = np.floor(row['start_time'] - 2)  # Subtract 2 seconds
            end_time = np.floor(start_time + 2 + row['total_distance'])

            # Create a mask for the rows in the heart rate data that fall within the interval
            mask = (data['elapsed_time'] >= start_time) & (data['elapsed_time'] <= end_time)
            # Use the mask to select the heart rate data and add it to the dictionary for the corresponding distance
            interval_data = data.loc[mask, 'heart_rate'].values

            if first_iteration and (len(interval_data) != (row['total_distance'] + 3)):
                # First interval needs 2 extra values before the start of the interval (start_time + 2 + row['total_distance'])
                duplicated_values = np.repeat(interval_data[0], 3)
                interval_data = np.concatenate((duplicated_values, interval_data))
                first_iteration = False

            distance_data[row['total_distance']]['heart_rate'].append(interval_data)

            # Add time and pace data to the dictionary
            distance_data[row['total_distance']]['time'].append(row['time_interval'])
            distance_data[row['total_distance']]['pace'].append(row['total_distance'] / row['time_interval'])
            distance_data[row['total_distance']]['rest'].append(row['rest'])

    # Convert the lists of arrays into matrices
    for distance in distances:
        if len(distance_data[distance]['heart_rate']) > 0:
            distance_data[distance]['heart_rate'] = np.vstack(distance_data[distance]['heart_rate'])

    return distance_data


import pandas as pd
from scipy.stats import zscore


import numpy as np
import pandas as pd

def correct_abnormal_rows(df, zscore_threshold=3.5, min_lengths=20):
    """
    Given a DataFrame with swimming split times, this function corrects abnormal rows by splitting
    them into two and adjusts the elapsed times and total distance.

    Parameters:
        df (DataFrame): Input DataFrame containing swimming splits data.
        zscore_threshold (float): outlier threshold
        min_lengths (int): min number of lengths in order to get reliable distribution to detect outliers

    Returns:
        DataFrame: Corrected DataFrame with updated total distance and speed.
    """

    # Define the strokes to iterate over
    strokes = ['freestyle', 'breaststroke', 'butterfly', 'backstroke']

    # Iterate over each stroke
    for stroke in strokes:
        # Filter the DataFrame for the current stroke
        stroke_data = df[df['swim_stroke'] == stroke]

        # Check if the stroke has sufficient data points
        if len(stroke_data) >= min_lengths:
            # Filter the stroke_data for active lengths
            active_stroke_data = stroke_data[stroke_data['length_type'] == 'active']

            # Calculate the z-scores for the elapsed times
            z_scores = np.abs((active_stroke_data['total_elapsed_time'] - active_stroke_data['total_elapsed_time'].mean()) / active_stroke_data['total_elapsed_time'].std())

            # Identify the indices of abnormal rows based on the z-score threshold
            abnormal_indices = active_stroke_data[z_scores > zscore_threshold].index

            # Iterate over the abnormal rows
            for index, row in df.loc[abnormal_indices].iterrows():
                print('correct row %f to %f' % (row['total_elapsed_time'], row['total_elapsed_time'] / 2))

                # Calculate the corrected time and speed
                corrected_time = row['total_elapsed_time'] / 2
                corrected_speed = row['avg_speed'] * 2
                lap_nr = row['lap_nr']
                length_nr = row['length_nr']

                # Update the original row with the corrected time and speed
                df.loc[index, 'total_elapsed_time'] = corrected_time
                df.loc[index, 'avg_speed'] = corrected_speed

                # Create a new row with corrected time and speed
                new_row = row.copy()
                new_row['total_elapsed_time'] = corrected_time
                new_row['avg_speed'] = corrected_speed
                new_row['length_nr'] = length_nr + 0.5
                new_row['edited'] = True

                # Check if the summary row exists for the current lap
                if sum((df['lap_nr'] == lap_nr) & df['length_nr'].isna()) == 0:
                    continue

                # Update the total distance in the summary row
                summary_idx = df[(df['lap_nr'] == lap_nr) & df['length_nr'].isna()].index[0]
                df.loc[summary_idx, 'total_distance'] += 25.0

                # Append the new row to the DataFrame
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).sort_values(by=['lap_nr', 'length_nr']).reset_index(drop=True)

    return df


def find_fastest_segments(df, distances=[25, 50, 100, 200, 300, 400]):
    """
    Find the fastest time for each of 25, 50, 100, 200, 300, and 400m
    in a dataframe of swimming lap times.

    Parameters:
        df (DataFrame): The swimming data with columns 'lap_nr', 'total_elapsed_time',
                        'total_distance', 'swim_stroke', 'length_nr', 'length_type'.
        distances (list):

    Returns:
        dict: Fastest time for each of 25, 50, 100, 200, 300, and 400m.
    """

    best_times = {i: {'time': np.inf, 'lap': None, 'all_lengths': None, 'pace': None} for i in distances}

    for lap in df.loc[
        (df['length_type'] == 'active') & (df['swim_stroke'] != 'drill'), 'lap_nr'].unique():
        lap_data = df[(df['lap_nr'] == lap) & (df['length_type'] == 'active')].sort_values(
            by='length_nr')
        all_lengths = lap_data['length_nr'].tolist()

        for distance in distances:
            if len(lap_data) < distance // 25:
                continue

            for i in range(0, len(lap_data) - (distance // 25) + 1):
                segment = lap_data.iloc[i:i + (distance // 25)]
                segment_time = segment['total_elapsed_time'].sum()

                if segment_time < best_times[distance]['time']:
                    best_times[distance]['time'] = segment_time
                    best_times[distance]['pace'] = segment_time*100/distance
                    best_times[distance]['lap_times'] = lap_data['total_elapsed_time']
                    best_times[distance]['all_lengths'] = np.arange(i, i + (distance // 25))

    # Replace infinite values with np.nan
    for distance in distances:
        if best_times[distance]['time'] == np.inf:
            best_times[distance]['time'] = np.nan

    return best_times

