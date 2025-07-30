"""
Utility functions for plot handling and highlighting movement regions in time series data.
Available Functions
-------------------
[Private]
_handle_plot(...): Handles the display and saving of matplotlib plots based on user-defined options.Handles the display
                and saving of matplotlib plots based on user-defined options.
_highlight_movement_regions(...): Highlights time intervals for each activity class (Sitting, Standing, Walking)
    using distinct colors.
"""


# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import os

# internal imports
from constants import ACTIVITY , TIME

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _handle_plot(save_dir:str, show=True, save=False, filename="plot.png")-> None:
    """
    Handles the display and saving of matplotlib plots based on user-defined options.

    This utility function centralizes logic for whether a plot should be shown on screen,
    saved to disk, or both. If saving is enabled, the function ensures the output directory exists
    and stores the plot using the specified filename.

    :param save_dir: String specifying the directory path where the plot should be saved
                     if `save=True`. The directory is created if it doesn't exist.

    :param show: Boolean indicating whether to display the plot interactively on screen.
                 If False, the plot is closed after saving. Default is True.

    :param save: Boolean indicating whether to save the plot as an image file. Default is False.


    :param filename: String specifying the name of the image file to save, including the extension
                     (e.g., "my_plot.png"). Only relevant if `save=True`.
                     Default is "plot.png".
    """
    if save:

        print(f"Saving plot to: {os.path.join(save_dir, filename)}")  # <--- debug

        # Create the output directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        # Save the current figure to the specified path
        plt.savefig(os.path.join(save_dir, filename))
    if show:
        plt.show()
    else:
        plt.close()

def _highlight_movement_regions(data, alpha=0.3):
    """
    Highlights time intervals for each activity class (Sitting, Standing, Walking)
    using distinct colors. Adds a single legend entry per activity.

    Assumes:
        - ACTIVITY column exists with codes: 0 = Sitting, 1 = Standing, 2 = Walking
        - TIME column exists for time axis

    :param data: pandas DataFrame with 't' and ACTIVITY columns.
    :param alpha: Transparency level for highlights (default = 0.3).
    """
    if ACTIVITY not in data.columns or TIME not in data.columns or data.empty:
        return

    # Define color and label for each activity
    activity_colors = {
        0: ('Sitting', 'blue'),
        1: ('Standing', 'orange'),
        2: ('Walking', 'green')
    }

    # Track whether label was already added
    label_added = {key: False for key in activity_colors.keys()}

    for activity_code, (label, color) in activity_colors.items():
        is_active = data[ACTIVITY] == activity_code
        change_points = is_active.ne(is_active.shift()).cumsum()
        groups = data[is_active].groupby(change_points)

        for _, group in groups:
            start_time = group[TIME].iloc[0]
            end_time = group[TIME].iloc[-1]
            current_label = label if not label_added[activity_code] else None
            plt.axvspan(start_time, end_time, color=color, alpha=alpha, label=current_label)
            label_added[activity_code] = True

    # Combine legend entries and remove duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

