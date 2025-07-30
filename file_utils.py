"""
This module provides utility functions to handle sensor data directories,
particularly selecting the earliest time-recorded folder within a given path.

Available Functions
-------------------
[Public]
find_hour_folder(...): Returns the subfolder with the earliest timestamp (HH-MM-SS format).
------------------

"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from datetime import datetime

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def get_hour_folder(path):
    time_folders = []
    for f in os.listdir(path):
        try:
            datetime.strptime(f, "%H-%M-%S")
            time_folders.append(f)
        except ValueError:
            continue
    if not time_folders:
        return None
    latest = min(time_folders, key=lambda x: datetime.strptime(x, "%H-%M-%S"))
    return os.path.join(path, latest)