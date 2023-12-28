from utils import write_datasets, dataset_pipeline, get_features_and_labels, load_and_test_model, get_landmarks
from consts import *
from utils_autoLandNet import test_AutoLandNet

FEATURES_CSV = os.path.join(DATA, "dataset.csv")
LABELS_XLSX  = os.path.join(DATA, "labels.xlsx")
LABELS_CSV   = os.path.join(DATA, "labels.csv")

UNPROCESSED_VIDEOS = TRAIN_UNPROCESSED_VIDEOS
PROCESSED_VIDEOS   = TRAIN_PROCESSED_VIDEOS

LABELS_ARGS   = (LABELS_XLSX, LABELS_CSV)
GAIT_ARGS     = (UNPROCESSED_VIDEOS, PROCESSED_VIDEOS)
FEATURES_ARGS = (PROCESSED_VIDEOS, FEATURES_CSV)

'''
####### EVAL #######
VFEATURES_CSV = os.path.join(DATA, "vdataset.csv")
VLABELS_CSV   = os.path.join(DATA, "vlabels.csv")
'''


def main():
    write_datasets(LABELS_ARGS, GAIT_ARGS, FEATURES_ARGS)
    print("videos processed!")

    dataset_pipeline(FEATURES_CSV)
    features, labels = get_features_and_labels(FEATURES_CSV, LABELS_CSV)
    landmarks = get_landmarks(features)
    print("features and landmarks processed!")

    load_and_test_model(features, labels, LAN_PATH, LAN_JSON)

    ###### EVALUATION AUTOLANDNET ######

    test_AutoLandNet(landmarks, labels)


if __name__ == "__main__":
    main()    
