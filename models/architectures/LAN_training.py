from models.architectures.landsAffectiveNet import ComplexEmotionsRegressor
from models.architectures.LAN_training_utils import train_simple, train_complex, train_medium
from utils import get_features_and_labels, get_landmarks, get_affectives, get_emotion
from consts import TRAIN_DATASET, TRAIN_LABELS_CSV_OUT
import os

features, labels = get_features_and_labels(TRAIN_DATASET, TRAIN_LABELS_CSV_OUT)
landmarks        = get_landmarks(features)
affectives       = get_affectives(features)
#emotion         = get_emotion(labels, "angry")
num_sequences    = len(landmarks)


#_________________________________________ simple model
simple_absolute_best = 1000000000000
simple_pos = -1
simple_losses = []

for i in range(15):
    print(i)

    history, model = train_simple(
        landmarks_units=256,
        epochs=200,
        lands=landmarks,
        learning_rate= 1e-3,
        activation="tanh",
        checkpoint_path = os.path.join("../../data", f"simple_best_weights{i}.h5"),
        json_path = os.path.join("../../data", "simple.json")
    )
    # Get the validation loss values
    validation_loss = history.history['val_loss']
    best_validation_losses = sorted(validation_loss)[:5]
    best = best_validation_losses[0]
    simple_losses.append(best)
    
    print("Best 5 Validation Loss Values:")
    for j, loss in enumerate(best_validation_losses):
        print(f"Epoch {j+1}: {loss}")

    if best < simple_absolute_best:
        simple_absolute_best = best
        simple_pos           = i

#_________________________________________ medium model
medium_absolute_best = 1000000000000
medium_pos = -1
medium_losses = []

for i in range(15):
    print(i)

    history, model = train_medium(
        landmarks_units=256,
        epochs=300,
        learning_rate= 1e-3,
        dropout=0.2,
        activation="tanh",
        checkpoint_path = os.path.join("../../data", f"medium_300_{i}.h5"),
        lands=landmarks,
        json_path = os.path.join("../../data", "medium.json")
    )
    # Get the validation loss values
    validation_loss = history.history['val_loss']
    best_validation_losses = sorted(validation_loss)[:5]
    best = best_validation_losses[0]
    medium_losses.append(best)
    
    print("Best 5 Validation Loss Values:")
    for j, loss in enumerate(best_validation_losses):
        print(f"Epoch {j+1}: {loss}")

    if best < medium_absolute_best:
        medium_absolute_best = best
        medium_pos           = i

#_________________________________________ complex
compl_absolute_best = 1000000000000
compl_pos = -1
compl_losses = []

for i in range(15):
    print(i)

    history, model = train_complex(
        landmarks_units=256,
        affective_units=88,
        epochs=300,
        learning_rate= 1e-3,
        dropout=0.2,
        activation="tanh",
        checkpoint_path = os.path.join("../../data", f"compl_300_{i}.h5"),
        lands=landmarks,
        affs=affectives,
        json_path = os.path.join("../../data", "complex.json"),
    )
    # Get the validation loss values
    validation_loss = history.history['val_loss']
    best_validation_losses = sorted(validation_loss)[:5]
    best = best_validation_losses[0]
    compl_losses.append(best)
    
    print("Best 5 Validation Loss Values:")
    for j, loss in enumerate(best_validation_losses):
        print(f"Epoch {j+1}: {loss}")

    if best < compl_absolute_best:
        compl_absolute_best = best
        compl_pos           = i

#_________________________________________ printing results __________________________
for i, l in enumerate(simple_losses):
    print(f"{i}, {l}")
avgS = sum(simple_losses)/len(simple_losses)
print(f"land best is {simple_absolute_best} at {simple_pos} --- avg: {avgS}")


for i, l in enumerate(medium_losses):
    print(f"{i}, {l}")
avgM = sum(medium_losses)/len(medium_losses)
print(f"land best is {medium_absolute_best} at {medium_pos} --- avg: {avgM}")


for i, l in enumerate(compl_losses):
    print(f"{i}, {l}")
avgC = sum(compl_losses)/len(compl_losses)
print(f"land best is {compl_absolute_best} at {compl_pos} --- avg: {avgC}")