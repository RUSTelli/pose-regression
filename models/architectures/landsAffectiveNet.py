from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, BatchNormalization, Dropout, Dense, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# landmarks and affective feature
class ComplexEmotionsRegressor:
    INPUT_SHAPE_LANDMARKS = (50, 66)
    INPUT_SHAPE_AFFECTIVES = (50, 22)

    def __init__(self, model=None):
        self.model = model

    def build_and_compile_model(self, learning_rate=1e-3, single_emotion_mode=False, landmarks_units=64,
                                affectives_units=32, checkpoint_path='best_weights.h5', dropout=0.1, activation="tanh"):
        self.landmarks_units = landmarks_units
        self.affectives_units = affectives_units
        self.dropout = dropout
        self.checkpoint_path = checkpoint_path
        self.activation = activation

        # Define the Landmarks encoder architecture
        landmarks_encoder = Input(shape=ComplexEmotionsRegressor.INPUT_SHAPE_LANDMARKS)
        encoded_sequence_landmarks = BatchNormalization()(landmarks_encoder)
        encoded_sequence_landmarks = LSTM(units=self.landmarks_units, activation=self.activation,
                                          return_sequences=True)(encoded_sequence_landmarks)
        encoded_sequence_landmarks = BatchNormalization()(encoded_sequence_landmarks)
        encoded_sequence_landmarks = Dropout(self.dropout)(encoded_sequence_landmarks)
        encoded_video_landmarks = LSTM(units=self.landmarks_units, activation=self.activation, return_sequences=False)(
            encoded_sequence_landmarks)

        affectives_encoder = Input(shape=ComplexEmotionsRegressor.INPUT_SHAPE_AFFECTIVES)
        encoded_sequence_affectives = BatchNormalization()(affectives_encoder)
        encoded_sequence_affectives = LSTM(units=self.affectives_units, activation=self.activation,
                                           return_sequences=True)(encoded_sequence_affectives)
        encoded_sequence_affectives = BatchNormalization()(encoded_sequence_affectives)
        encoded_sequence_affectives = Dropout(self.dropout)(encoded_sequence_affectives)
        encoded_video_affectives = LSTM(units=self.affectives_units, activation=self.activation,
                                        return_sequences=False)(encoded_sequence_affectives)

        # defining the input for the regression head
        merged_outputs = concatenate([encoded_video_landmarks, encoded_video_affectives])
        dense_units = 1 if single_emotion_mode else 4
        regression_output = Dense(units=dense_units, activation='linear')(merged_outputs)

        model = Model(inputs=[landmarks_encoder, affectives_encoder], outputs=regression_output)
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='mean_squared_error')
        if self.model == None: self.model = model

    def train(self, landmarks, affectives, labels, epochs=10, batch_size=32, validation_split=0.2, patience=10,
              json_path="complex.json"):
        if self.model is None:
            raise ValueError("Model is not built. Call build_model() first.")

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_loss', save_best_only=True)

        history = self.model.fit(
            x=[landmarks, affectives],
            y=labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint]
        )

        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)

        return history

    def test(self, landmarks, affectives, labels):
        if self.model is None:
            raise ValueError("Model is not built. Call build_and_compile_model() first.")

        # Implement testing logic using self.model
        predictions = self.model.predict([landmarks, affectives])

        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")


    def plot_loss(self, history):
        if history is None:
            raise ValueError("History is None. Call train() to get training history.")

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @classmethod
    def load_best_weights(cls, weights_path: str, json_path: str, single_emotion_mode=False):
        # getting architecture
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(weights_path)
        return cls(model)


# only landmarks
class MediumEmotionRegressor:
    INPUT_SHAPE_LANDMARKS = (50, 66)

    def __init__(self, model=None):
        self.model = model

    def build_and_compile_model(self, learning_rate=1e-3, single_emotion_mode=False, landmarks_units=64,
                                checkpoint_path='best_weights.h5', dropout=0.1, activation="tanh"):
        self.landmarks_units = landmarks_units
        self.dropout = dropout
        self.checkpoint_path = checkpoint_path
        self.activation = activation

        # Define the Landmarks encoder architecture
        landmarks_encoder = Input(shape=ComplexEmotionsRegressor.INPUT_SHAPE_LANDMARKS)
        encoded_sequence_landmarks = BatchNormalization()(landmarks_encoder)
        encoded_sequence_landmarks = LSTM(units=self.landmarks_units, activation=self.activation,
                                          return_sequences=True)(encoded_sequence_landmarks)
        encoded_sequence_landmarks = BatchNormalization()(encoded_sequence_landmarks)
        encoded_sequence_landmarks = Dropout(self.dropout)(encoded_sequence_landmarks)
        encoded_video_landmarks = LSTM(units=self.landmarks_units, activation=self.activation, return_sequences=False)(
            encoded_sequence_landmarks)

        # defining the input for the regression head
        dense_units = 1 if single_emotion_mode else 4
        regression_output = Dense(units=dense_units, activation='linear')(encoded_video_landmarks)

        model = Model(inputs=[landmarks_encoder], outputs=regression_output)
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='mean_squared_error')
        self.model = model

    def train(self, landmarks, labels, epochs=10, batch_size=32, validation_split=0.2, patience=10,
              json_path="medium.json"):
        if self.model is None:
            raise ValueError("Model is not built. Call build_model() first.")

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_loss', save_best_only=True)

        history = self.model.fit(
            x=[landmarks],
            y=labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint]
        )

        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)

        return history

    def test(self, landmarks, affectives, labels):
        if self.model is None:
            raise ValueError("Model is not built. Call build_and_compile_model() first.")

        # Implement testing logic using self.model
        predictions = self.model.predict([landmarks])

        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2): {r2}")

        plt.scatter(labels, predictions)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.show()

    def plot_loss(self, history):
        if history is None:
            raise ValueError("History is None. Call train() to get training history.")

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @classmethod
    def load_best_weights(cls, weights_path: str, json_path: str, single_emotion_mode=False):
        # getting architecture
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(weights_path)
        return cls(model)


# only landmarks 1 lstm
class SimpleEmotionRegressor:
    INPUT_SHAPE_LANDMARKS = (50, 66)

    def __init__(self, model=None):
        self.model = model

    def build_and_compile_model(self, learning_rate=1e-3, single_emotion_mode=False, landmarks_units=64,
                                checkpoint_path='best_weights.h5', activation="tanh"):
        self.landmarks_units = landmarks_units
        self.checkpoint_path = checkpoint_path
        self.activation = activation

        # Define the Landmarks encoder architecture
        landmarks_encoder = Input(shape=ComplexEmotionsRegressor.INPUT_SHAPE_LANDMARKS)
        encoded_sequence_landmarks = BatchNormalization()(landmarks_encoder)
        encoded_sequence_landmarks = LSTM(units=self.landmarks_units, activation=self.activation,
                                          return_sequences=False)(encoded_sequence_landmarks)

        # defining the input for the regression head
        dense_units = 1 if single_emotion_mode else 4
        regression_output = Dense(units=dense_units, activation='linear')(encoded_sequence_landmarks)

        model = Model(inputs=[landmarks_encoder], outputs=regression_output)
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='mean_squared_error')
        self.model = model

    def train(self, landmarks, labels, epochs=10, batch_size=32, validation_split=0.2, patience=10,
              json_path="simple.json"):
        if self.model is None:
            raise ValueError("Model is not built. Call build_model() first.")

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_loss', save_best_only=True)

        history = self.model.fit(
            x=[landmarks],
            y=labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint]
        )

        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)

        return history

    def test(self, landmarks, affectives, labels):
        if self.model is None:
            raise ValueError("Model is not built. Call build_and_compile_model() first.")

        # Implement testing logic using self.model
        predictions = self.model.predict([landmarks])

        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2): {r2}")

        plt.scatter(labels, predictions)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.show()

    def plot_loss(self, history):
        if history is None:
            raise ValueError("History is None. Call train() to get training history.")

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @classmethod
    def load_best_weights(cls, weights_path: str, json_path: str, single_emotion_mode=False):
        # getting architecture
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(weights_path)
        return cls(model)