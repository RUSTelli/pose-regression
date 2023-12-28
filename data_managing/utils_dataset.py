import csv
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from data_managing.consts import VIDEO_NAME_COLUMN, MAX_GAIT_LEN

def _drop_head_landmarks_and_save(features_csv: str) -> None:
    df = pd.read_csv(features_csv)
    # Remove the first 44 columns (excluding the first column 'video_name')
    columns_to_remove = df.columns[1:45]
    df = df.drop(columns=columns_to_remove)
    df.to_csv(features_csv, index=False)

def _drop_visibility_and_save(features_csv: str) -> None:
    df = pd.read_csv(features_csv)
    # Identify columns starting with 'v'
    columns_to_drop = [col for col in df.columns if col.startswith('v')]
    # Drop the identified columns
    df = df.drop(columns=columns_to_drop)
    # Save the modified DataFrame back to the original CSV file
    df.to_csv(features_csv, index=False)

def _split_features(features: List[Any]) -> tuple:
    landmarks = np.array(features[:66])
    distances = np.array(features[66:66+11])
    angles = np.array(features[77:77+7])
    areas = np.array(features[84:84+4])
    return landmarks, distances, angles, areas

######################################################
def _split_features2(features: List[Any]) -> tuple:
    landmarks  = np.array(features[:66])
    affectives = np.array(features[66:])
    return landmarks, affectives

def _group_features_by_video(features_csv: str) -> Dict[str, Dict[str, List[np.ndarray]]]:
    data = pd.read_csv(features_csv)
    grouped_sequences = {}

    # Iterate through each row in the DataFrame
    for _, row in data.iterrows():
        sequence_name = row[VIDEO_NAME_COLUMN]
        features = row.drop(VIDEO_NAME_COLUMN).tolist()
        landmarks, distances, angles, areas = _split_features(features)
        
        if sequence_name not in grouped_sequences:
            grouped_sequences[sequence_name] = {
                "landmarks": [],
                "distances": [],
                "angles"   : [],
                "areas"    : [],
            }
        
        grouped_sequences[sequence_name]["landmarks"].append(landmarks)
        grouped_sequences[sequence_name]["distances"].append(distances)
        grouped_sequences[sequence_name]["angles"].append(angles)
        grouped_sequences[sequence_name]["areas"].append(areas)
    
    return grouped_sequences

######################################################
def _group_features_by_video2(features_csv: str) -> Dict[str, Dict[str, List[np.ndarray]]]:
    data = pd.read_csv(features_csv)
    grouped_sequences = {}

    # Iterate through each row in the DataFrame
    for _, row in data.iterrows():
        sequence_name = row[VIDEO_NAME_COLUMN]
        features = row.drop(VIDEO_NAME_COLUMN).tolist()
        landmarks, affectives = _split_features2(features)
        
        if sequence_name not in grouped_sequences:
            grouped_sequences[sequence_name] = {
                "landmarks": [],
                "affectives": [],
            }
        
        grouped_sequences[sequence_name]["landmarks"].append(landmarks)
        grouped_sequences[sequence_name]["affectives"].append(affectives)
    return grouped_sequences

def _zero_pad(sequence: List[np.ndarray]) -> List[np.ndarray]:
    required_pad = MAX_GAIT_LEN - len(sequence)
    pad_elem_len = sequence[0].shape[0]
    for _ in range(required_pad):
        sequence.append(np.zeros(pad_elem_len))
    return sequence

def get_padded_features(features_csv: str) -> Dict[str, Dict[str, np.ndarray]]:
    padded_features = {}
    grouped_features = _group_features_by_video(features_csv)
    
    for sequence_name, features in grouped_features.items():
        pad_landmarks = _zero_pad(features["landmarks"])
        pad_distances = _zero_pad(features["distances"])
        pad_angles    = _zero_pad(features["angles"])
        pad_areas     = _zero_pad(features["areas"])
        
        padded_features[sequence_name] = {
            "landmarks": np.array(pad_landmarks),
            "distances": np.array(pad_distances),
            "angles"   : np.array(pad_angles),
            "areas"    : np.array(pad_areas),
        }
    
    return padded_features

######################################################
def get_padded_features2(features_csv: str) -> Dict[str, Dict[str, np.ndarray]]:
    padded_features = {}
    grouped_features = _group_features_by_video2(features_csv)
    
    for sequence_name, features in grouped_features.items():
        pad_landmarks  = _zero_pad(features["landmarks"])
        pad_affectives = _zero_pad(features["affectives"])
        
        padded_features[sequence_name] = {
            "landmarks" : np.array(pad_landmarks),
            "affectives": np.array(pad_affectives)
        }
    
    return padded_features

def features_pipeline(features_csv: str) -> None:
    _drop_head_landmarks_and_save(features_csv)
    _drop_visibility_and_save(features_csv)

def labels_preprocessing(in_labels_xlsx: str, out_labels_csv: str) -> None:
    df = pd.read_excel(in_labels_xlsx, index_col=0)

    # elimino la riga contenente le domande per poter effettuare le operazioni
    # l'ordine delle emozioni è: Happy, Angry, Sad, Neutral
    df.drop('Question', axis=0, inplace=True)

    # effettuo la media per colonna per ogni video (quindi per ciascuna delle 4 emozioni)
    df_mean = df.mean()

    # divido ciascuna media ottenuta per 5 così da avere la percentuale di
    # appartenenza a ciascuna delle emozioni per ogni video
    df_percent = df_mean.div(5)

    # approssimo i valori ottenuti mantenendo solo le prime due cifre significative dopo la virgola
    srs_percent_rounded = round(df_percent, 2)

    # converto da series a dataframe
    df_percent_rounded = srs_percent_rounded.to_frame()

    # converto in csv e xlsx e lo salvo nella dir out
    df_percent_rounded.to_excel(in_labels_xlsx, sheet_name='labels')
    df_percent_rounded.to_csv(out_labels_csv, header=False)

def get_labels(labels_csv: str):
    labels = []

    with open(labels_csv, 'r') as file:
        csv_reader = csv.reader(file)
        values = []

        for row in csv_reader:
            value = float(row[1])
            values.append(value)

            if len(values) == 4:
                labels.append(tuple(values))
                values = []
    return np.array(labels)