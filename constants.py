# ------------------------------------------------------------------------------------------------------------------- #
# sensor constants
# ------------------------------------------------------------------------------------------------------------------- #
# definition of valid sensors (for now only phone sensors)
ACC = 'ACC'
GYR = 'GYR'
MAG = 'MAG'
ROT = 'ROT'

VALID_SENSORS = [ACC, GYR, MAG, ROT]
IMU_SENSORS = [ACC, GYR, MAG]

# mapping of valid sensors to sensor filename
SENSOR_MAP = {ACC: 'ANDROID_ACCELEROMETER',
              GYR: 'ANDROID_GYROSCOPE',
              MAG: 'ANDROID_MAGNETIC_FIELD',
              ROT: 'ANDROID_ROTATION_VECTOR'}

# sampling rate
FS = 100

# Accelerometer axes
ACC_Y_COL = 'y_ACC'
ACC_Z_COL = 'z_ACC'


# Acceleration components in the world reference frame
WORLD_ACC = ['a_rX', 'a_rY', 'a_rZ']

# Rotational difference data
ROT_COL = 'rot_diff'

ACC_ENVELOPE = 'acc_envelope'

ACTIVITY = 'activity'
BLOCK_ID = 'block_id'

ANGLE = 'angle_rad'
DISP_X = 'dx'
DISP_Y = 'dy'
TIME = 't'
TRAJECTORY_X = 'x'
TRAJECTORY_Y = 'y'
ANGULAR_VEL = 'angular_velocity'

WALKING = 2
STANDING = 1
SITTING = 0

WALKING_NAME = 'Walking'
SITTING_NAME = 'Sitting'
STANDING_NAME = 'Standing'

PROPORTIONS = 'Proportions'

