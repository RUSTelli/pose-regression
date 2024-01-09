import os.path

from models.architectures.autoencoder import Autoencoder
from models.architectures.autoLandsNet import LSTMSVRDecoder
from utils import get_features_and_labels, get_landmarks, get_affectives, get_emotion
from consts import TRAIN_DATASET, TRAIN_LABELS_CSV_OUT, CHECKPOINT_PATH
from utils import get_features_and_labels
from consts import *
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from keras.models import Sequential, load_model
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# TRAINING

LABELS_CSV   = os.path.join(DATA, "labels.csv")
FEATURES_CSV = os.path.join(DATA, "dataset.csv")


'''
# EVALUATION
VLABELS_CSV   = os.path.join(DATA, "vlabels.csv")
VFEATURES_CSV = os.path.join(DATA, "vdataset.csv")
'''

'''
features, labels = get_features_and_labels(FEATURES_CSV, LABELS_CSV)
landmarks        = get_landmarks(features)
num_sequences    = len(landmarks)
'''


latent_dim = 100
lstm_weights_path = 'encoder_weights.h5'

'''
# affective
# landmarks        = get_affectives(features)
# num_sequences    = len(landmarks)

encoding_dim = 100

X_train, X_test = train_test_split(landmarks, test_size=0.3, random_state=42)

####### TRAINING MODEL #######
# Create an Autoencoder instance with BatchNormalization and Dropout
autoencoder = Autoencoder(encoding_dim, checkpoint_path="encoder_weights.h5")
autoencoder.train(X_train, num_epochs=600, batch_size=num_sequences, early_stopping_patience=100)

# Plot the training and validation loss
autoencoder.plot_loss_history()

# Save encoder weights
autoencoder.save_encoder_weights("encoder_weights.h5")

######## TESTING #########
# Load encoder in another class
model = Autoencoder.load_best_weights('encoder_weights.h5')
X_test_reshaped = X_test.reshape(X_test.shape[0], 50, 66) # (50,66) for landmarks
model.test_model(X_test_reshaped)
'''
'''
######### REGRESSOR #######

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(landmarks, labels, test_size=0.2, random_state=42)


lstmsvr_decoder = LSTMSVRDecoder(latent_dim, lstm_weights_path, num_emotions=4)
lstmsvr_decoder.train_decoders(landmarks, labels)
lstmsvr_decoder.save_decoder_weights('decoder_weights')

'''


######### TEST ON THE FOUR DECODERS ##########
def test_AutoLandNet(landmarks, labels):
    latent_dim = 100
    lstm_weights_path = LSTMENCODER_PATH

    # List to store MSE and MAE for each decoder
    mse_list = []
    mae_list = []

    for decoder_index in range(4):
        new_lstm_svr_decoder = LSTMSVRDecoder(latent_dim=latent_dim, lstm_weights_path=lstm_weights_path,
                                              num_emotions=4)
        new_lstm_svr_decoder.load_decoder_weights(filepath_prefix=os.path.join("data", WEIGHTS_DIR, 'decoder_weights'))

        encoded_data_new = new_lstm_svr_decoder.encode(landmarks)
        decoded_data_new = new_lstm_svr_decoder.decode(encoded_data_new)

        mse_test = mean_squared_error(labels, np.array(decoded_data_new).T)
        mae_test = mean_absolute_error(labels, np.array(decoded_data_new).T)

        mse_list.append(mse_test)
        mae_list.append(mae_test)

        # Plotting for each emotion
        for i in range(labels.shape[1]):
            plt.figure(figsize=(8, 4))  # Create a new figure for each emotion
            plt.scatter(range(len(labels)), labels[:, i], label=f'Actual - Emotion {i + 1}')
            plt.scatter(range(len(decoded_data_new[i])), decoded_data_new[i], label=f'Predicted - Emotion {i + 1}',
                        marker='x')

            plt.legend()
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title(f'Decoder {decoder_index + 1} - Emotion {i + 1}')
            plt.show()

    # Print overall MSE and MAE for each decoder
    for decoder_index, (mse, mae) in enumerate(zip(mse_list, mae_list), 1):
        print(f'Decoder {decoder_index} - Mean Squared Error on Test Set: {mse}')
        print(f'Decoder {decoder_index} - Mean Absolute Error on Test Set: {mae}')

'''
############### EVALUATION ##############
new_lstm_svr_decoder = LSTMSVRDecoder(latent_dim=latent_dim, lstm_weights_path=lstm_weights_path, num_emotions=4)
new_lstm_svr_decoder.load_decoder_weights("decoder_weights")
encoded_data_new = new_lstm_svr_decoder.encode(landmarks)
decoded_data_new = new_lstm_svr_decoder.decode(encoded_data_new)
mse_test = mean_squared_error(labels, np.array(decoded_data_new).T)
print(f'Mean Squared Error on Test Set: {mse_test}')

# plot the scatter plot for each of the four emotion on test data
plt.figure(figsize=(15, 8))
for i in range(labels.shape[1]):
    plt.subplot(2, 2, i + 1)  # Create a subplot for each emotion
    plt.scatter(range(len(labels)), labels[:, i], label=f'Actual - Emotion {i + 1}')
    plt.scatter(range(len(decoded_data_new[i])), decoded_data_new[i], label=f'Predicted - Emotion {i + 1}', marker='x')

    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'Emotion {i + 1}')

plt.tight_layout()
plt.show()
'''

'''
# Train the SVR decoder
lstmsvr_decoder.train_decoders(X_train, y_train)

# save weights
lstmsvr_decoder.save_model('full_model_weights.h5')

encoded_data_train = lstmsvr_decoder.encode(X_train)
encoded_data_test = lstmsvr_decoder.encode(X_test)

# "Decode" the encoded test data (in this case, it's the encoded data itself)
decoded_data_test = lstmsvr_decoder.decode(encoded_data_test)

# Evaluate the performance
mse_test = mean_squared_error(y_test, np.array(decoded_data_test).T)
print(f'Mean Squared Error on Test Set: {mse_test}')

# plot the scatter plot for each of the four emotion on test data
plt.figure(figsize=(15, 8))
for i in range(y_test.shape[1]):
    plt.subplot(2, 2, i + 1)  # Create a subplot for each emotion
    plt.scatter(range(len(y_test)), y_test[:, i], label=f'Actual - Emotion {i + 1}')
    plt.scatter(range(len(decoded_data_test[i])), decoded_data_test[i], label=f'Predicted - Emotion {i + 1}', marker='x')

    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'Emotion {i + 1}')

plt.tight_layout()
plt.show()
'''





