"""

This function provides a visualization of time-based activity metrics.
Available Functions
-------------------
[Public]
plot_metrics_per_day(...): Plots a stacked bar chart for each day of acquisition, showing proportions of walking, sitting, and standing.

-------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
from .plot_utils import _handle_plot
from collections import defaultdict
from constants import PROPORTIONS
# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
TOTAL_DURATION = 'total_duration'

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def plot_metrics_per_day(all_metrics: dict, show: bool, save: bool, save_dir: str) -> None:
    """
    Plots a stacked bar chart for each day of acquisition, showing proportions of walking, sitting, and standing.

    :param all_metrics: Dictionary with session IDs as keys and activity metrics as values.
                        Expected format: {"group_subject_date_time": {"movement": {...}, "sitting": {...}, "standing": {...}}}
    :param show: Whether to display the plot.
    :param save: Whether to save the plot to disk.
    :param save_dir: Directory path to save plots.
    """
    # Group metrics by subject
    subject_data = defaultdict(dict)  # subject -> {date -> {movement, sitting, standing}}

    for session_id, metrics in all_metrics.items():
        try:
            parts = session_id.split("_")
            subject = parts[1]
            date = parts[2]

            subject_data[subject][date] = metrics
        except IndexError:
            print(f"[!] Failed to parse session_id: {session_id}")
            continue

    # Plot per subject
    for subject, day_metrics in subject_data.items():
        dates = sorted(day_metrics.keys())
        walking_props, sitting_props, standing_props = [], [], []

        for date in dates:
            m = day_metrics[date]
            proportions = m.get(PROPORTIONS, {
                "walking_proportion": 0.0,
                "sitting_proportion": 0.0,
                "standing_proportion": 0.0
            })

            walking_props.append(proportions["walking_proportion"])
            sitting_props.append(proportions["sitting_proportion"])
            standing_props.append(proportions["standing_proportion"])

        # Plot stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_positions = list(range(len(dates)))

        ax.bar(bar_positions, walking_props, color='#1f77b4', label='Walking')
        ax.bar(bar_positions, sitting_props, bottom=walking_props, color='#ff7f0e', label='Sitting')
        bottom_2 = [w + s for w, s in zip(walking_props, sitting_props)]
        ax.bar(bar_positions, standing_props, bottom=bottom_2, color='#2ca02c', label='Standing')

        ax.set_xticks(bar_positions)
        ax.set_xticklabels(dates, rotation=45)
        ax.set_ylabel("Proportion of Time")
        ax.set_title(f"Activity Distribution per Day for {subject}")
        ax.legend()
        plt.tight_layout()

        filename = f"{subject}_activity_distribution_per_day.png"
        _handle_plot(save_dir=save_dir, show=show, save=save, filename=filename)


