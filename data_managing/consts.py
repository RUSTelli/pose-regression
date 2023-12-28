from enum import Enum
from mediapipe.python.solutions.pose import PoseLandmark as LANDMARKS


# list of landmarks to exclude from the drawing (those regarding the face)
excluded_landmarks = [
    LANDMARKS.LEFT_EYE,
    LANDMARKS.RIGHT_EYE,
    LANDMARKS.LEFT_EYE_INNER,
    LANDMARKS.RIGHT_EYE_INNER,
    LANDMARKS.LEFT_EAR,
    LANDMARKS.RIGHT_EAR,
    LANDMARKS.LEFT_EYE_OUTER,
    LANDMARKS.RIGHT_EYE_OUTER,
    LANDMARKS.NOSE,
    LANDMARKS.MOUTH_LEFT,
    LANDMARKS.MOUTH_RIGHT
]


class MEDIAN_LANDMARKS(Enum):
    ROOT = (LANDMARKS.RIGHT_HIP, LANDMARKS.LEFT_HIP)
    NECK = (LANDMARKS.RIGHT_SHOULDER, LANDMARKS.LEFT_SHOULDER)


DISTANCES = {
    (MEDIAN_LANDMARKS.ROOT, MEDIAN_LANDMARKS.NECK),
    # fingers distances
    (LANDMARKS.RIGHT_INDEX, MEDIAN_LANDMARKS.ROOT),
    (LANDMARKS.RIGHT_INDEX, MEDIAN_LANDMARKS.NECK),
    (LANDMARKS.RIGHT_INDEX, LANDMARKS.LEFT_INDEX),
    (LANDMARKS.LEFT_INDEX, MEDIAN_LANDMARKS.ROOT),
    (LANDMARKS.LEFT_INDEX, MEDIAN_LANDMARKS.NECK),
    # toes distances
    (LANDMARKS.RIGHT_FOOT_INDEX, MEDIAN_LANDMARKS.ROOT),
    (LANDMARKS.RIGHT_FOOT_INDEX, MEDIAN_LANDMARKS.NECK),
    (LANDMARKS.RIGHT_FOOT_INDEX, LANDMARKS.LEFT_FOOT_INDEX),
    (LANDMARKS.LEFT_FOOT_INDEX, MEDIAN_LANDMARKS.ROOT),
    (LANDMARKS.LEFT_FOOT_INDEX, MEDIAN_LANDMARKS.NECK),
}

ANGLES = {
    (LANDMARKS.RIGHT_SHOULDER, LANDMARKS.LEFT_SHOULDER, MEDIAN_LANDMARKS.ROOT),
    # arms angles
    (LANDMARKS.RIGHT_INDEX, LANDMARKS.LEFT_INDEX, MEDIAN_LANDMARKS.ROOT),
    (LANDMARKS.RIGHT_SHOULDER, LANDMARKS.RIGHT_ELBOW, LANDMARKS.RIGHT_INDEX),
    (LANDMARKS.LEFT_SHOULDER, LANDMARKS.LEFT_ELBOW, LANDMARKS.LEFT_INDEX),
    # legs angles
    (LANDMARKS.RIGHT_FOOT_INDEX, LANDMARKS.LEFT_FOOT_INDEX, MEDIAN_LANDMARKS.ROOT),
    (LANDMARKS.RIGHT_HIP, LANDMARKS.RIGHT_KNEE, LANDMARKS.RIGHT_FOOT_INDEX),
    (LANDMARKS.LEFT_HIP, LANDMARKS.LEFT_KNEE, LANDMARKS.LEFT_FOOT_INDEX),
}

AREAS = {
    (LANDMARKS.RIGHT_SHOULDER, LANDMARKS.LEFT_SHOULDER, MEDIAN_LANDMARKS.ROOT),
    (LANDMARKS.RIGHT_INDEX, LANDMARKS.LEFT_INDEX, MEDIAN_LANDMARKS.ROOT),
    (LANDMARKS.RIGHT_INDEX, LANDMARKS.LEFT_INDEX, MEDIAN_LANDMARKS.NECK),
    (LANDMARKS.RIGHT_FOOT_INDEX, LANDMARKS.LEFT_FOOT_INDEX, MEDIAN_LANDMARKS.ROOT),
}

AXIS = ['x', 'y', 'z', 'v']
MAX_GAIT_LEN = 50
VIDEO_NAME_COLUMN = "Video_name"