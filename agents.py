from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import os


class Agent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.states, self.actions, self.rewards = [], [], []
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        if self.state_size > 16:
            model.add(Conv2D(32, (7, 7), activation='relu', input_shape=self.state_size))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (7, 7), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (7, 7), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (7, 7), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(24, activation='relu'))
            model.add(Dense(24, activation='relu'))
        else:
            model.add(Dense(24, activation='relu', input_shape=(self.state_size,), name='input_layer'))
            model.add(Dense(24, activation='relu'))

        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

        if os.path.exists('./save_model/pruning_agent.h5'):
            self.model.load_weights('./save_model/pruning_agent.h5')

        return model

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def get_action(self, state):
        action = self.model.predict(state)[0]
        return action

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

        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]
        self.model.fit(update_inputs, advantages, epochs=3, verbose=1)
        self.states, self.actions, self.rewards = [], [], []
