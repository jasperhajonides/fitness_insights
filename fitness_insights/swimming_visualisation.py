import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit

def plot_all_reps_swimset(df_laps: pd.DataFrame, df_hr: pd.DataFrame, ax: plt.Axes):
    """Plot the swim data for all repetitions in a swimset.

    Parameters:
        df_laps (pd.DataFrame): DataFrame containing lap data with columns: 'lap_nr', 'sub_sport', 'total_distance', 'total_elapsed_time', 'swim_stroke', 'start_time'.
        df_hr (pd.DataFrame): DataFrame containing heart rate data with columns: 'elapsed_time', 'heart_rate'.
        ax (plt.Axes): The matplotlib axes to plot the data on.

    Returns:
        None
    """
    color_palette = {'freestyle': 'blue', 'backstroke': 'orange',
                     'butterfly': 'red', 'breaststroke': 'purple',
                     'drill':'green', 'mixed':'gray'}

    for lp in range(df_laps.lap_nr.max()):
        group = df_laps.loc[(df_laps['lap_nr'] == lp)]
        avg = df_laps.loc[(df_laps['lap_nr'] == lp) & (df_laps['sub_sport'] == 'lap_swimming')]
        avg = avg.copy()
        avg.at[:, 'pace'] = avg['total_distance'].values[0] / avg['total_elapsed_time'].values[0] * 100

        if None in group.swim_stroke.to_list():
            continue

        ax.bar(avg['start_time'].values[0],
               avg['pace']/100,
               width=group['total_elapsed_time'],
               align='edge',
               color=color_palette[avg.swim_stroke.to_list()[-1]],
               alpha=0.4)

    ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Average Speed')
    ax.set_title('Bar Chart of Average Speed by Elapsed Time')
    ax.set_ylim([0, 2.1])

    ax_ = ax.twinx()

    ax_.plot(df_hr['elapsed_time'], df_hr['heart_rate'], 'red')
    ax_.plot(df_hr['elapsed_time'], df_hr['heart_rate'], 'red', linewidth=6, alpha=0.2)
    ax_.plot(df_hr['elapsed_time'], df_hr['heart_rate'], 'red', linewidth=7, alpha=0.05)

    ax_.set_ylim([0, 200])


def plot_avg_paces(df_laps: pd.DataFrame, ax: plt.Axes, ylim=[60, 110]):
    """Plot the average paces based on total distance.

    Parameters:
        df_laps (pd.DataFrame): DataFrame containing lap data with columns: 'sub_sport', 'total_distance', 'total_elapsed_time'.
        ax (plt.Axes): The matplotlib axes to plot the data on.
        ylim (list, optional): The y-axis limits for the plot. Default is [60, 110].

    Returns:
        None
    """
    data = df_laps.loc[df_laps['sub_sport'] == 'lap_swimming']

    data_filtered = data[data['total_distance'].notna()]
    data_filtered = data_filtered[data_filtered['total_distance'] != 0]

    data_filtered['pace'] = (data_filtered['total_elapsed_time'] * 100) / data_filtered['total_distance']

    grouped_data = data_filtered.groupby('total_distance')['pace'].median()

    ax.scatter(data_filtered['total_distance'], data_filtered['pace'], color='gray', alpha=0.5, label='Individual Pace')
    ax.scatter(grouped_data.index, grouped_data.values, color='white', alpha=0.8, s=100, edgecolor='black', label='Average Pace')

    # Fit the logarithmic function to the grouped data
    params, _ = curve_fit(logarithmic_func, grouped_data.index, grouped_data.values)
    x_fit = np.linspace(min(np.array(grouped_data.index)), max(np.array(grouped_data.index)), 100)
    y_fit = logarithmic_func(x_fit, params[0], params[1])
    ax.plot(x_fit, y_fit, color='r')

    ax.set_title('Pace (seconds per 100m) vs Total Distance')
    ax.set_xlabel('Total Distance')
    ax.set_ylabel('Pace (seconds per 100m)')
    ax.legend()
    ax.set_xlim([0, 650])
    ax.set_ylim(ylim)

def logarithmic_func(x, a, b):
    """Logarithmic function used for curve fitting.

    Parameters:
        x (float): Input value.
        a (float): First parameter.
        b (float): Second parameter.

    Returns:
        float: The output value of the logarithmic function.
    """
    return a * np.log(x) + b



def plot_hr_intervals(distance_data: dict, ax: plt.Axes, distance: int = 200):
    """Plot heart rate intervals for a given distance.

    Parameters:
        distance_data (dict): A dictionary containing dataframes for each distance with heart rate, time, pace, and rest data.
        ax (plt.Axes): The matplotlib axes to plot the data on.
        distance (int, optional): The distance for which heart rate intervals will be plotted. Default is 200.

    Returns:
        None
    """
    paces = distance_data[distance]['pace']
    paces = [1 / i * 100 for i in paces]

    norm = plt.Normalize(75, 95)
    cmap = cm.get_cmap('magma_r')
    color_map = cmap(norm(paces))

    for i, interval in enumerate(distance_data[distance]['heart_rate']):
        ln = ax.plot(interval, color=color_map[i], label=paces[i])

    ax.plot(np.mean(distance_data[distance]['heart_rate'], axis=0), color='white', linewidth=3)

    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    ax.set_title(f"Heart Rate for {distance}m Intervals")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart Rate (BPM)")
