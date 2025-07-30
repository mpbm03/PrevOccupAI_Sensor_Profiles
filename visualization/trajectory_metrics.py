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
from datetime import datetime
from matplotlib.ticker import FuncFormatter


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
    subject_data = defaultdict(dict)  # subject -> {date -> metrics}
    subject_to_group = dict()  # subject -> group

    for session_id, metrics in all_metrics.items():
        try:
            parts = session_id.split("_")
            group = parts[0]
            subject = parts[1]
            date_str = parts[2]  # "2022-06-20"

            subject_data[subject][date_str] = metrics

            # Guarda o grupo para o sujeito (supõe grupo constante por sujeito)
            if subject not in subject_to_group:
                subject_to_group[subject] = group

        except IndexError:
            print(f"[!] Failed to parse session_id: {session_id}")
            continue

    for subject, day_metrics in subject_data.items():
        dates_sorted = sorted(day_metrics.keys())
        walking_props, sitting_props, standing_props = [], [], []
        date_labels = []

        for date_raw in dates_sorted:
            m = day_metrics[date_raw]
            proportions = m.get(PROPORTIONS, {
                "walking_proportion": 0.0,
                "sitting_proportion": 0.0,
                "standing_proportion": 0.0
            })

            walking_props.append(proportions["walking_proportion"])
            sitting_props.append(proportions["sitting_proportion"])
            standing_props.append(proportions["standing_proportion"])

            # Formata a data com dia da semana
            date_obj = datetime.strptime(date_raw, "%Y-%m-%d")
            date_formatted = date_obj.strftime("%d/%m/%Y (%A)")
            date_labels.append(date_formatted)

        group = subject_to_group.get(subject, "Unknown Group")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        positions = list(range(len(dates_sorted)))

        ax.bar(positions, walking_props, color='#1f77b4', label='Walking')
        ax.bar(positions, sitting_props, bottom=walking_props, color='#ff7f0e', label='Sitting')
        bottom_2 = [w + s for w, s in zip(walking_props, sitting_props)]
        ax.bar(positions, standing_props, bottom=bottom_2, color='#2ca02c', label='Standing')

        ax.set_xticks(positions)
        ax.set_xticklabels(date_labels, rotation=45, ha='right')
        ax.set_ylabel("Percentage of Time (%)")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

        ax.set_title(f"Group: {group} | Activity Distribution per Day for {subject}")
        ax.legend()
        plt.tight_layout()

        filename = f"{group}_{subject}_activity_distribution_per_day.png"
        _handle_plot(save_dir=save_dir, show=show, save=save, filename=filename)
