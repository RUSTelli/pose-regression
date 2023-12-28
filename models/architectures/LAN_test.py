from models.architectures.landsAffectiveNet import ComplexEmotionsRegressor, MediumEmotionRegressor, SimpleEmotionRegressor
from utils import get_features_and_labels, get_landmarks, get_affectives
import os

VAL_DATASET = os.path.join("../../data", "Vdataset.csv")
VAL_LABELS  = os.path.join("../../data", "Vlabels.csv")

features, labels = get_features_and_labels(VAL_DATASET, VAL_LABELS)
landmarks        = get_landmarks(features)
affectives       = get_affectives(features)

COMPLEX_WEIGHTS = os.path.join("../../data", "c.h5")
COMPLEX_JSON    = os.path.join("../../data", "complex.json")
cmpl            = ComplexEmotionsRegressor.load_best_weights(COMPLEX_WEIGHTS, COMPLEX_JSON)

MEDIUM_WEIGHTS  = os.path.join("../../data", "m.h5")
MEDIUM_JSON     = os.path.join("../../data", "medium.json")
mdm             = MediumEmotionRegressor.load_best_weights(MEDIUM_WEIGHTS, MEDIUM_JSON)

SIMPLE_WEIGHTS  = os.path.join("../../data", "s.h5")
SIMPLE_JSON     = os.path.join("../../data", "simple.json")
smpl            = SimpleEmotionRegressor.load_best_weights(SIMPLE_WEIGHTS, SIMPLE_JSON)


models = [
    cmpl, 
    mdm, 
    smpl,
]

for m in models:
    m.test(landmarks, affectives, labels)