import os

DATA                = "data"
WEIGHTS_DIR         = "autolandnet_weights"
WEIGHTS_DIR_LAN     = "landsaffectivenet_weights"
CHECKPOINT_PATH     = os.path.join(DATA, "best_weights.h5")

######################  A U T O L A N D N E T #####################

LSTMENCODER_PATH = os.path.join(DATA, WEIGHTS_DIR, "encoder_weights.h5")
DECODER_WEIGHT_0 = os.path.join(DATA, WEIGHTS_DIR, 'decoder_weights_decoder_0.joblib')
DECODER_WEIGHT_1 = os.path.join(DATA, WEIGHTS_DIR, 'decoder_weights_decoder_1.joblib')
DECODER_WEIGHT_2 = os.path.join(DATA, WEIGHTS_DIR, 'decoder_weights_decoder_2.joblib')
DECODER_WEIGHT_3 = os.path.join(DATA, WEIGHTS_DIR, 'decoder_weights_decoder_3.joblib')

######################  L A N D S A F F E C T I V E N E T #####################

LAN_PATH = os.path.join(DATA, WEIGHTS_DIR_LAN, "c.h5")
LAN_JSON = os.path.join(DATA, WEIGHTS_DIR_LAN, "complex.json")

##########################  T R A I N #############################
### GAIT ###
TRAIN_UNPROCESSED_VIDEOS   = os.path.join(DATA, "raw_videos")
TRAIN_PROCESSED_VIDEOS     = os.path.join(DATA, "processed_videos")
### DATASET ###
TRAIN_DATASET              = os.path.join(DATA, "dataset.csv")
TRAIN_LABELS_XLSX          = os.path.join(DATA, "labels.xlsx")
TRAIN_LABELS_CSV_OUT       = os.path.join(DATA, "labels.csv")
#####################  V A L I D A T I O N ########################
### GAIT ###
TRAIN_UNPROCESSED_VIDEOS   = os.path.join(DATA, "raw_videos")
TRAIN_PROCESSED_VIDEOS     = os.path.join(DATA, "processed_videos")
### DATASET ###
TRAIN_DATASET              = os.path.join(DATA, "dataset.csv")
TRAIN_LABELS_XLSX          = os.path.join(DATA, "labels.xlsx")
TRAIN_LABELS_CSV_OUT       = os.path.join(DATA, "labels.csv")
##########################  T E S T  ##############################
TEST_LABELS_XLSX          = "" # tutor
TEST_LABELS_CSV           = "test_label.csv"

TEST_UNPROCESSED_VIDEOS   = "" # tutor
TEST_PROCESSED_VIDEOS     = ""

TEST_LANDMARKS_DATASET    = ""

TEST_LABELS               = ""
TEST_LABELS_REARRANGED    = ""