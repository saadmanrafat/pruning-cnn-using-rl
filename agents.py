from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras import Input
from keras.optimizers import Adam
from keras import Input
import numpy as np
import os
import keras


class Agent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.state_dim = state_size[-1]
        self.action_size = action_size
        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.states, self.actions, self.rewards = [], [], []
        self.model = self._build_model()
        if os.path.exists('pruning_agent.h5'):
            self.model.load_weights('./saved_model/pruning_agent.h5')

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (7, 7), activation='relu', padding="same", input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Conv2D(64, (7, 7), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Conv2D(64, (7, 7), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Conv2D(64, (7, 7), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Flatten())
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        return self.model.predict(state)

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train_model(self):
        episode_length = len(self.states)
        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        height, width, feature_map = self.state_size
        update_inputs = np.zeros((episode_length, height, width, feature_map))
        advantages = np.zeros((episode_length, self.action_size))
        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]
        print(update_inputs)
        print(advantages)
        self.model.fit(update_inputs, advantages, epochs=10, verbose=1, callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1)
        ])
        self.states, self.actions, self.rewards = [], [], []
