from os.path import abspath, dirname, join

# CSV and HDF files
PROJECT_ROOT_DIR = dirname(abspath(__file__))  # Directory containing this file
DATASET_PATH = join(PROJECT_ROOT_DIR, "datasets")
OUTPUT_PATH = join(PROJECT_ROOT_DIR, "raw_data.h5")

ACCELEROMETER_PATH = join(DATASET_PATH, "Accelerometer.csv")
ACT_INST_PATH = join(DATASET_PATH, "Activity_instances.csv")
AROUSAL_QUESTIONNAIRE_PATH = join(DATASET_PATH, "Arousal_Questionnaires.csv")
MARKERS_PATH = join(DATASET_PATH, "Custom_markers.csv")
EDA_PATH = join(DATASET_PATH, "Eda.csv")
ENVIRONMENT_PATH = join(DATASET_PATH, "Environment.csv")
GYROSCOPE_PATH = join(DATASET_PATH, "Gyroscope.csv")
HEART_RATE_PATH = join(DATASET_PATH, "Heart_rate.csv")
NOISE_LEVEL_PATH = join(DATASET_PATH, "Noise_level.csv")
QUESTIONNAIRE_PATH = join(DATASET_PATH, "Questionnaires.csv")
QUICK_NOTES_PATH = join(DATASET_PATH, "Quick_Notes.csv")
SKIN_TEMP_PATH = join(DATASET_PATH, "Skin_temperature.csv")

ALL_CSV_PATHS = (ACCELEROMETER_PATH, ACT_INST_PATH, AROUSAL_QUESTIONNAIRE_PATH,
                 MARKERS_PATH, EDA_PATH, ENVIRONMENT_PATH, GYROSCOPE_PATH,
                 HEART_RATE_PATH, NOISE_LEVEL_PATH, QUESTIONNAIRE_PATH,
                 QUICK_NOTES_PATH, SKIN_TEMP_PATH)