from models.architectures.autoencoder import Autoencoder
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
import joblib


class LSTMSVRDecoder:
    INPUT_SHAPE_LANDMARKS = (50, 66)

    def __init__(self, latent_dim, lstm_weights_path, num_emotions):
        self.latent_dim = latent_dim
        self.lstm_weights_path = lstm_weights_path
        self.num_emotions = num_emotions

        # Build the LSTM Encoder
        self.encoder = self.build_encoder()

        # Build the Multi-output SVR Decoder
        self.decoders = self.build_decoders()

    def build_encoder(self):
        encoder = Sequential()
        encoder.add(LSTM(self.latent_dim, input_shape=LSTMSVRDecoder.INPUT_SHAPE_LANDMARKS, name='encoder_lstm', trainable=False))
        encoder.load_weights(self.lstm_weights_path, by_name=True)
        return encoder

# FUNCT THAT USES RANDOM FOREST REGRESSOR
    def build_decoders(self, use_grid_search=True):
        decoders = []
        for _ in range(self.num_emotions):

            if use_grid_search:
                # Set up a parameter grid for grid search
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                # Instantiate a Random Forest Regressor
                base_regressor = RandomForestRegressor()

                # Perform grid search
                grid_search = GridSearchCV(base_regressor, param_grid, cv=3, scoring='neg_mean_squared_error')
                decoders.append(grid_search)
            else:
                # If not using grid search, simply use the base regressor
                decoders.append(RandomForestRegressor())

        return decoders

    def save_decoder_weights(self, filepath_prefix):
        for i, decoder in enumerate(self.decoders):
            filename = f"{filepath_prefix}_decoder_{i}.joblib"
            joblib.dump(decoder, filename)

    def load_decoder_weights(self, filepath_prefix):
        for i in range(self.num_emotions):
            filename = f"{filepath_prefix}_decoder_{i}.joblib"
            self.decoders[i] = joblib.load(filename)
    '''
        # FUNCT THAT USES SVR AS REGRESSOR 
    def build_decoders(self):
        decoders = []
        for _ in range(self.num_emotions):
            decoder = SVR(kernel='rbf')
            decoders.append(decoder)
        return decoders
    '''
    # FUNCT TO TRAIN WITH CROSS VALIDATION
    def train_decoders(self, x_data, y_data, use_grid_search=True, num_folds=2):
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        for train_index, val_index in kf.split(x_data):
            x_train, x_val = x_data[train_index], x_data[val_index]
            y_train, y_val = y_data[train_index], y_data[val_index]

            encoded_data = self.encode(x_train)

            for i, decoder in enumerate(self.decoders):
                if use_grid_search:
                    # Set up a parameter grid for grid search
                    param_grid = {
                        'n_estimators': [10, 50, 100],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                    # Instantiate a Random Forest Regressor
                    base_regressor = RandomForestRegressor()

                    # Perform grid search
                    grid_search = GridSearchCV(base_regressor, param_grid, cv=3, scoring='neg_mean_squared_error')
                    grid_search.fit(encoded_data.reshape(-1, self.latent_dim), y_train[:, i])

                    # Get the best hyperparameters found by grid search
                    best_params = grid_search.best_params_
                    print(f'Best hyperparameters for Emotion {i + 1}: {best_params}')

                    # Create a new instance of the regressor with the best hyperparameters
                    best_regressor = RandomForestRegressor(**best_params)
                    best_regressor.fit(encoded_data.reshape(-1, self.latent_dim), y_train[:, i])

                    # Replace the grid search object with the new regressor
                    self.decoders[i] = best_regressor
                else:
                    # Fit without grid search
                    decoder.fit(encoded_data.reshape(-1, self.latent_dim), y_train[:, i])

            # evaluate the performance on the validation set here
            encoded_data_val = self.encode(x_val)
            decoded_data_val = self.decode(encoded_data_val)
            mse_val = mean_squared_error(y_val, np.array(decoded_data_val).T)
            print(f'Mean Squared Error on Validation Set: {mse_val}')
            # plot the scatter plot for each of the four emotion on test data
            plt.figure(figsize=(15, 8))
            for i in range(y_val.shape[1]):
                plt.subplot(2, 2, i + 1)  # Create a subplot for each emotion
                plt.scatter(range(len(y_val)), y_val[:, i], label=f'Actual - Emotion {i + 1}')
                plt.scatter(range(len(decoded_data_val[i])), decoded_data_val[i],
                            label=f'Predicted - Emotion {i + 1}', marker='x')

                plt.legend()
                plt.xlabel('Sample Index')
                plt.ylabel('Value')
                plt.title(f'Emotion {i + 1}')

            plt.tight_layout()
            plt.show()

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, encoded_data):
        decoded_data = []
        for decoder in self.decoders:
            decoded_data.append(decoder.predict(encoded_data))
        return decoded_data

    def save_model(self, filepath):
        self.encoder.save(filepath)

    def load_model(self, filepath):
        loaded_model = Sequential()
        loaded_model.add(LSTM(self.latent_dim, input_shape=LSTMSVRDecoder.INPUT_SHAPE_LANDMARKS, name='encoder_lstm', trainable=False))
        loaded_model.load_weights(filepath, by_name=True)
        self.encoder = loaded_model
        self.decoders = self.build_decoders()