from models.architectures.landsAffectiveNet import ComplexEmotionsRegressor, MediumEmotionRegressor, SimpleEmotionRegressor
from utils import get_features_and_labels, get_landmarks, get_affectives
from consts import TRAIN_DATASET, TRAIN_LABELS_CSV_OUT

features, labels = get_features_and_labels(TRAIN_DATASET, TRAIN_LABELS_CSV_OUT)
landmarks        = get_landmarks(features)
affectives       = get_affectives(features)
num_sequences    = len(landmarks)

LANDMARKS  = [64, 128,  256]
AFFECTIVES = [22, 44,   88]
EPOCHS     = [100, 300, 500]
L_RATE     = [1e-3, 1e-4, 1e-5]
DROPOUT    = [0.1, 0.15, 0.2]
ACTIVATION = ["tanh", "relu"]

def train_complex(landmarks_units, affective_units, epochs, learning_rate, dropout, activation, checkpoint_path, lands, affs, json_path):
    # Instantiate and then build the EmotionRecognizer class
    emotion_recognizer = ComplexEmotionsRegressor()

    emotion_recognizer.build_and_compile_model(
        learning_rate = learning_rate, 
        single_emotion_mode=False,
        landmarks_units=landmarks_units, 
        affectives_units=affective_units, 
        checkpoint_path=checkpoint_path,
        dropout=dropout,
        activation=activation
    )

    # Train the model with early stopping and save the best weights
    history = emotion_recognizer.train(
        lands, 
        affs, 
        labels, 
        epochs=epochs, 
        batch_size=num_sequences, 
        validation_split=0.2, 
        patience=epochs // 5,
        json_path=json_path,
    )

    return history, emotion_recognizer

def train_medium(landmarks_units, epochs, learning_rate, dropout, activation, checkpoint_path, lands, json_path):
    # Instantiate and then build the EmotionRecognizer class
    emotion_recognizer = MediumEmotionRegressor()

    emotion_recognizer.build_and_compile_model(
        learning_rate = learning_rate, 
        single_emotion_mode=False,
        landmarks_units=landmarks_units,
        checkpoint_path=checkpoint_path,
        dropout=dropout,
        activation=activation
    )

    # Train the model with early stopping and save the best weights
    history = emotion_recognizer.train(
        lands,
        labels, 
        epochs=epochs, 
        batch_size=num_sequences, 
        validation_split=0.2, 
        patience=epochs // 5,
        json_path=json_path,
    )

    return history, emotion_recognizer

def train_simple(landmarks_units, epochs, learning_rate, activation, checkpoint_path, lands, json_path):
    # Instantiate and then build the EmotionRecognizer class
    emotion_recognizer = SimpleEmotionRegressor()

    emotion_recognizer.build_and_compile_model(
        learning_rate = learning_rate, 
        single_emotion_mode=False,
        landmarks_units=landmarks_units,
        checkpoint_path=checkpoint_path,
        activation=activation
    )

    # Train the model with early stopping and save the best weights
    history = emotion_recognizer.train(
        lands,
        labels, 
        epochs=epochs, 
        batch_size=num_sequences, 
        validation_split=0.2, 
        patience=epochs // 5,
        json_path=json_path,
    )

    return history, emotion_recognizer

def grid_search():
    histories = []
    count = 0

    for landmark in LANDMARKS:
        for affective in AFFECTIVES:
            for epoch in EPOCHS:
                for l_rate in L_RATE:
                    for dropout in DROPOUT:
                        for activation in ACTIVATION:
                            param = {
                                "landmark"  : landmark,
                                "affective" : affective,
                                "epoch"     : epoch,
                                "l_rate"    : l_rate,
                                "dropout"   : dropout,
                                "activation": activation,
                                "loss"      : []
                            }

                            history1, _ = train_complex(landmark, affective, epoch, l_rate, dropout, activation)
                            history2, _ = train_complex(landmark, affective, epoch, l_rate, dropout, activation)
                            history3, _ = train_complex(landmark, affective, epoch, l_rate, dropout, activation)

                            count = count + 3
                            print(count)

                            # Get the validation loss values
                            l1 = sorted(history1.history['val_loss'])[0]
                            l2 = sorted(history2.history['val_loss'])[0]
                            l3 = sorted(history3.history['val_loss'])[0]
                            param["loss"].append((l1, l2, l3))
                            histories.append(param)

                            if count % 2 == 0:
                                top = sorted(histories, key=lambda x: sorted(x["loss"])[0])
                                for i, combination in enumerate(top):
                                    print(i, combination)
    return histories                    