# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import joblib
import tsfel
import os
import json

# internal imports
from constants import VALID_SENSORS, FS, TIME, STANDING, SITTING, WALKING_NAME, SITTING_NAME, STANDING_NAME, PROPORTIONS
from load_data import load_data_from_same_recording
from preprocess_data import data_preprocessing
from Trajectory import compute_trajectory
from metrics import analyze_total_movement , analyze_stationary_segments, calculate_activity_proportions
from visualization import generate_motion_visualizations, plot_metrics_per_day
from HAR import apply_classification_pipeline
from feature_extractor import extract_features
from file_utils import get_hour_folder


# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #

FOLDER_PATH = r"D:\Mariana\1º ano Mestrado - 2º semestre\Prevoccupai\data" # definition of folder_path
OUTPUT_JSON_PATH = r"D:\Mariana\1º ano Mestrado - 2º semestre\Prevoccupai\data\metrics"
OUTPUT_FILENAME = "all_metrics.json"
OUTPUT_PATH = os.path.join(OUTPUT_JSON_PATH, OUTPUT_FILENAME)

SCALAR_FIRST = False               # indicates whether quaternions are in scalar-first format (w, x, y, z) or scalar-last (x, y, z, w).
PADDING_TYPE = "same"              # padding which should be used to ensure that all sensors start and stop at the same time

WINDOW_SIZE_FE = 5                 # Window size for feature extraction or smoothing (samples)
WINDOW_SIZE_ROT = 2                # Window size for smoothing angle differences (seconds)
MOVING_WINDOW = 150                # Window size for smoothing acceleration data (samples)

MIN_WALK_DURATION_MIN = 1          # Minimum walking duration in minutes

PEAK_THRESHOLD = 0.15               # Percentage of the signal envelope's maximum used as the peak detection threshold
VALLEY_THRESHOLD = 0.15             # Percentage of the signal envelope's maximum used as the valley detection threshold
ANGULAR_VELOCITY_THRESHOLD = 120     # Angular velocity threshold (deg/s) above which steps are discarded (e.g., due to sudden spikes or noise)
ROTATION_THRESHOLD = 45             # Threshold for rotation-based event detection


SENSOR_AXIS_DICT = {'ACC': ['y']}  # Dictionary specifying which sensors and axes to plot

SUBPLOT_COLUMNS = 3                # Number of columns in subplot visualization

SHOW_PLOTS =False
SAVE_PLOTS = False
PLOTS_DIRECTORY = r"D:\Mariana\1º ano Mestrado - 2º semestre\Prevoccupai\data\plots"

PROB_THRESHOLD = 0.85 # threshold for probability thresholding
MIN_DURATIONS = {0: 20, 1: 30, 2: 5} # durations for 0 (sitting), 1 (standing), 2 (walking)

MODEL_SENSORS = 'x_ACC', 'y_ACC', 'z_ACC', 'x_GYR', 'y_GYR', 'z_GYR', 'x_MAG', 'y_MAG', 'z_MAG' #Sensors used to train the model


# load config file
cfg = tsfel.load_json("cfg_file_production_model.json")
# loading the classifier
model = joblib.load("HAR_model_500.joblib")


if __name__ == '__main__':
    if not (0 <= PEAK_THRESHOLD <= 1):
        raise ValueError(f"Invalid peak_threshold: {PEAK_THRESHOLD}. It must be between 0 and 1.")

    if not (0 <= VALLEY_THRESHOLD <= 1):
        raise ValueError(f"Invalid valley_threshold: {VALLEY_THRESHOLD}. It must be between 0 and 1.")

    # Dictionary to hold all processed metrics
    all_metrics = {}

    # Traverse each group (e.g., group1, group2)
    for group in os.listdir(FOLDER_PATH):
        group_path = os.path.join(FOLDER_PATH, group)
        if not os.path.isdir(group_path):
            continue

        # Look inside each category within the group (e.g., 'sensors', 'questionnaires')
        for category in os.listdir(group_path):
            category_path = os.path.join(group_path, category)

            # We are only interested in sensor data
            if category != "sensors":
                continue

            # Traverse each subject folder (e.g., LIBPhys #001)
            for subject in os.listdir(category_path):
                subject_path = os.path.join(category_path, subject)
                if not os.path.isdir(subject_path):
                    continue

                # Inside each subject, look into each date folder (e.g., 2022-06-20)
                for date_folder in os.listdir(subject_path):
                    date_path = os.path.join(subject_path, date_folder)
                    if not os.path.isdir(date_path):
                        continue

                    # From each date folder, find the most recent time folder (e.g., 11-20-19)
                    latest_hour_path = get_hour_folder(date_path)
                    if not latest_hour_path:
                        print(f"[!] No time folders in {date_path}")
                        continue

                    # Create a unique session identifier
                    session_id = f"{group}_{subject}_{date_folder}_{os.path.basename(latest_hour_path)}"
                    print(f"▶ Processing: {session_id}")

                    try:
                        # 1. Load and preprocess
                        raw_sensor_data = load_data_from_same_recording(latest_hour_path, VALID_SENSORS, FS, PADDING_TYPE)
                        data = data_preprocessing(raw_sensor_data, fs=FS, scalar_first=SCALAR_FIRST)

                        # 2. Extract features and classify
                        data, model_features = extract_features(data, cfg, model, MODEL_SENSORS, WINDOW_SIZE_FE, FS)
                        data, y_pred, y_pred_exp = apply_classification_pipeline(
                            data, model_features, model, w_size=WINDOW_SIZE_FE, fs=FS,
                            threshold=PROB_THRESHOLD, min_durations=MIN_DURATIONS)

                        # 3. Compute trajectory
                        data, valid_peaks, xs, ys, densities, lengths = compute_trajectory(
                            data, FS=FS, peak_threshold=PEAK_THRESHOLD, valley_threshold=VALLEY_THRESHOLD,
                            window_mov_average=MOVING_WINDOW, deriv_threshold=ANGULAR_VELOCITY_THRESHOLD)

                        # 4. Compute metrics
                        movement_metrics = analyze_total_movement(data, valid_peaks)
                        sitting_metrics = analyze_stationary_segments(data, rotation_threshold=ROTATION_THRESHOLD,
                                                                      window_s=WINDOW_SIZE_ROT, activity_code=SITTING, overlap=0.1)
                        standing_metrics = analyze_stationary_segments(data, rotation_threshold=ROTATION_THRESHOLD,
                                                                       window_s=WINDOW_SIZE_ROT, activity_code=STANDING,
                                                                       overlap=0.1)
                        proportions = calculate_activity_proportions(
                            movement_metrics,
                            sitting_metrics,
                            standing_metrics
                        )

                        # 5. (Optional) Visualization
                        if SHOW_PLOTS or SAVE_PLOTS:
                            generate_motion_visualizations(
                                data, valid_peaks, xs, ys, densities,
                                ANGULAR_VELOCITY_THRESHOLD,
                                movement_metrics, sitting_metrics, standing_metrics, lengths,
                                save=SAVE_PLOTS, save_dir=PLOTS_DIRECTORY
                            )
                        # 6. Save results
                        all_metrics[session_id] = {
                            WALKING_NAME: movement_metrics,
                            SITTING_NAME: sitting_metrics,
                            STANDING_NAME: standing_metrics,
                            PROPORTIONS: proportions
                        }

                    except Exception as e:
                        print(f"[Error] Failed to process {session_id}: {e}")

    # Optional: save metrics to disk
    # Create the directory if it doesn't exist
    os.makedirs(OUTPUT_JSON_PATH, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Done. Total sessions processed: {len(all_metrics)}")

    # Plot stacked bar chart per subject across days
    plot_metrics_per_day(
        all_metrics=all_metrics,
        show=True,
        save=True,
        save_dir=PLOTS_DIRECTORY
    )










