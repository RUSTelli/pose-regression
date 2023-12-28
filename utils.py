from typing import Dict
import numpy as np
from data_managing.utils_gait import extract_gaits_from_folder
from data_managing.utils_features import write_features_dataset
from data_managing.utils_dataset import features_pipeline, get_padded_features2, labels_preprocessing, get_labels
from models.architectures.landsAffectiveNet import ComplexEmotionsRegressor


def _manage_labels(in_labels_xlsx: str, out_labels_csv: str) -> None:
    labels_preprocessing(in_labels_xlsx, out_labels_csv)

def write_datasets(labels_args: tuple, gait_args: tuple, features_args: tuple) -> None:
    extract_gaits_from_folder(*gait_args)
    write_features_dataset(*features_args)
    _manage_labels(*labels_args)

def dataset_pipeline(features_csv: str) -> None:
    features_pipeline(features_csv)

def get_features_and_labels(features_csv: str, labels_csv: str) -> tuple:
    features =  get_padded_features2(features_csv)
    labels = get_labels(labels_csv)
    return features, labels

def get_landmarks(features: Dict):
    num_videos = len(features)
    landmarks = np.empty((num_videos, 50, 66), dtype=float)

    for video_index, video_features in enumerate(features.values()):
        lands = np.array(video_features["landmarks"])
        landmarks[video_index] = lands
    return landmarks

def get_emotion(labels, emotion: str):
    emotions_index = {
        "happy"   : 0,
        "angry"   : 1,
        "sad"     : 2,
        "neutral" : 3,
    }

    index = emotions_index[emotion]
    selected_component = labels[:, index]
    return selected_component.reshape(-1, 1)

def get_affectives(features: Dict):
    num_videos = len(features)
    affectives = np.empty((num_videos, 50, 22), dtype=float)

    for video_index, video_features in enumerate(features.values()):
        affects = np.array(video_features["affectives"])
        affectives[video_index] = affects
    return affectives


def load_and_test_model(features: Dict, labels: list, weights_path = "", json_path = "") -> None:
    landmarks  = get_landmarks(features)
    affectives = get_affectives(features)

    model = ComplexEmotionsRegressor.load_best_weights(weights_path, json_path)
    model.test(landmarks, affectives, labels)


