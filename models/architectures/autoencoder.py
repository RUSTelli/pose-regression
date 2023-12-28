import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Input, LSTM, BatchNormalization, Dropout
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam


class Autoencoder:
    INPUT_SHAPE_LANDMARKS = (50, 66)
    # INPUT_SHAPE_LANDMARKS = (50, 22) # affective feature

    def __init__(self, encoding_dim, checkpoint_path='encoder_weights.h5'):
        self.encoding_dim = encoding_dim
        self.checkpoint_path = checkpoint_path
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = self.build_autoencoder()

    # test aggiungendo strato LSTM
    def build_encoder(self):
        input_layer = Input(shape=Autoencoder.INPUT_SHAPE_LANDMARKS)
        input_layer = BatchNormalization()(input_layer)
        encoded = LSTM(self.encoding_dim, return_sequences=True, activation='relu')(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = LSTM(self.encoding_dim, return_sequences=True, activation='relu')(encoded)
        encoder = Model(input_layer, encoded)
        return encoder

    # test aggiungendo strato LSTM
    def build_decoder(self):
        input_layer = Input(shape=(Autoencoder.INPUT_SHAPE_LANDMARKS[0], self.encoding_dim))
        input_layer =  BatchNormalization()(input_layer)
        decoded = LSTM(self.encoding_dim, return_sequences=True, activation='relu')(input_layer)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = LSTM(Autoencoder.INPUT_SHAPE_LANDMARKS[1], return_sequences=True, activation='relu')(decoded)
        decoder = Model(input_layer, decoded)
        return decoder

    def build_autoencoder(self, learning_rate = 1e-3):
        input_layer = Input(shape=Autoencoder.INPUT_SHAPE_LANDMARKS)
        encoded = self.encoder(input_layer)
        decoded = self.decoder(encoded)
        autoencoder = Model(input_layer, decoded)
        opt = Adam(learning_rate=learning_rate)
        autoencoder.compile(optimizer=opt, loss='mse')
        return autoencoder

    def train(self, train_data, num_epochs, batch_size, early_stopping_patience=None):
        callbacks = []
        if early_stopping_patience is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True
            )
            checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_loss', save_best_only=True)
            callbacks.append(early_stopping)
            callbacks.append(checkpoint)

        # Store the training history
        self.history = self.autoencoder.fit(
                                        train_data,
                                        train_data,
                                        epochs=num_epochs,
                                        batch_size=batch_size,
                                        validation_split=0.2,
                                        callbacks=callbacks
                                      )

    def plot_loss_history(self):
        if self.history is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    def save_encoder_weights(self, filepath):
        self.encoder.save(filepath)

    def test_model(self, landmarks):
        if self.autoencoder is None:
            raise ValueError("Model is not built. Call build_and_compile_model() first.")

        #predictions = self.autoencoder.predict(landmarks)
        # print(predictions.shape)
        test_loss = self.autoencoder.evaluate(landmarks, landmarks)
        print(test_loss)

        # mse = mean_squared_error(landmarks, predictions)
        # mae = mean_absolute_error(landmarks, predictions)
        # r2 = r2_score(landmarks, predictions)
        # print(f"Mean Squared Error (MSE): {mse}")
        # print(f"Mean Absolute Error (MAE): {mae}")
        # print(f"R-squared (R2): {r2}")


    @classmethod
    def load_best_weights(cls, weights_path: str):
        mdl = cls(encoding_dim=100)
        mdl.build_autoencoder(learning_rate=1e-3)
        print(mdl.autoencoder.summary())
        mdl.autoencoder.load_weights(weights_path, by_name=True)
        return mdl

