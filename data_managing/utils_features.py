import mediapipe as mp
import math as math
import numpy as np
import itertools
import cv2
import csv
import os
from data_managing.consts import DISTANCES, ANGLES, AREAS, MEDIAN_LANDMARKS, AXIS, VIDEO_NAME_COLUMN


def _tuple_avg(tuple1: tuple, tuple2: tuple) -> tuple:
    return tuple((a + b) / 2 for a, b in zip(tuple1, tuple2))

def _axis(landmark) -> tuple:
    return landmark.x, landmark.y, landmark.z, landmark.visibility

def _distance(pt1, pt2) -> float:
    x1, y1, z1, v1 = pt1
    x2, y2, z2, v2 = pt2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return dist

def _angle(vector_a: tuple, vector_b: tuple, vector_c: tuple) -> float:
    # Extract the first three components of the input tuples
    vector_a = np.array(vector_a[:3])
    vector_b = np.array(vector_b[:3])
    vector_c = np.array(vector_c[:3])

    # Calculate the dot products of AB and BC
    dot_product_ab = np.dot(vector_a, vector_b)
    dot_product_bc = np.dot(vector_b, vector_c)

    # Calculate the magnitudes of AB and BC
    magnitude_ab = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    magnitude_bc = np.linalg.norm(vector_b) * np.linalg.norm(vector_c)

    # Calculate the cosine of the angle using the dot products and magnitudes
    cosine_angle = (dot_product_ab * dot_product_bc) / (magnitude_ab * magnitude_bc)

    # Calculate the angle in radians
    angle_rad = math.acos(cosine_angle)

    # Convert the angle to degrees
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def _area(vectors):
    # Extract the first two components of each vector in 'vectors'
    updated_vectors = [vector[:2] for vector in vectors]
    vectors = updated_vectors
    total_area = 0
    for i, v1 in enumerate(vectors):
        for v2 in vectors[i+1:]:
            cross_product = np.cross(v1, v2)
            total_area += 0.5 * abs(cross_product)
    return total_area

def _get_coordiantes(landmarks: list, points) -> tuple:
    coords = []

    for landmark in points:
        if isinstance(landmark, MEDIAN_LANDMARKS):
            lnd1, lnd2 = landmark.value
            cd1, cd2 = _axis(landmarks[lnd1]), _axis(landmarks[lnd2])
            coordinates = _tuple_avg(cd1, cd2)
        else:
            coordinates = _axis(landmarks[landmark])
        coords.append(coordinates)
    return tuple(coords)

def _get_affective_features(landmarks: list) -> list:
    affective_features = []

    for points in DISTANCES:
        cd1, cd2 = _get_coordiantes(landmarks, points)
        dist = _distance(cd1, cd2)
        affective_features.append(dist)

    for points in ANGLES:
        cd1, cd2, cd3 = _get_coordiantes(landmarks, points)
        angle = _angle(cd1, cd2, cd3)
        affective_features.append(angle)

    for points in AREAS:
        coordinates = _get_coordiantes(landmarks, points)
        area = _area(coordinates)
        affective_features.append(area)

    return affective_features

def _get_header() -> list:
    video_name       = [VIDEO_NAME_COLUMN]
    landmarks_header = [f'{s}_{i}' for i in range(33) for s in AXIS]
    distances_header = [f'd_{i}'   for i in range(11)]
    angles_header    = [f'an_{i}'  for i in range(7)]
    areas_header     = [f'ar_{i}'  for i in range(4)]
    
    header_elements = [
        video_name,
        landmarks_header,
        distances_header,
        angles_header,
        areas_header,
    ]
    return list(itertools.chain.from_iterable(header_elements))
    
def _get_video_features(video_file: str) -> list:
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        video_features = []
        cap = cv2.VideoCapture(video_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                # Make detection
                results = pose.process(frame)
                if results.pose_landmarks:
                    frame_landmarks = results.pose_landmarks.landmark

                    frame_landmarks_coords = [_axis(landmark) for landmark in frame_landmarks]
                    # unpacking frame_landmarks_coords (list of tuples -> list of values)
                    plain_coords = list(itertools.chain.from_iterable(frame_landmarks_coords))
                    frame_affective_features = _get_affective_features(frame_landmarks)
                    plain_coords.extend(frame_affective_features)
                    video_features.append(plain_coords)
            else:
                break
        cap.release()
        return video_features
    
def write_features_dataset(videos_path: str, out_dataset_path: str) -> None:
    with open(out_dataset_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(_get_header())

        for filename in os.listdir(videos_path):
            if filename.endswith(".mp4"):
                video_file_path = os.path.join(videos_path, filename)
                # Write each frame's features as a separate row
                for frame_features in _get_video_features(video_file_path):
                    row = list(itertools.chain.from_iterable([[filename], frame_features]))
                    csv_writer.writerow(row) 