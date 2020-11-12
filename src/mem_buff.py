import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Conv2D

from tensorflow.keras.applications import ResNet50

class MemBuff:
    def __init__(self, obs_dim, n_actions):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.model = self.build_network()
        
    def build_network(self):
        # resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        inp = Input(shape=self.obs_dim)
        # x = Conv2D(16, (3,3), activation="relu")(inp)
        # x = Conv2D(32, (3,3), activation="relu")(x)
        # x = Flatten()(x)
        
        x = Dense(128, activation="relu")(inp)
        x = Dense(64, activation="relu")(x)
        # x = LSTM(128, activation="relu")(x)
        # x = Flatten()(x)
        output = Dense(self.n_actions, activation="linear")(x)
        
        model = Model(inputs=inp, outputs=output)
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        
        return model
        
    def predict(self, obs):
        pred = self.model.predict(obs)
        return pred