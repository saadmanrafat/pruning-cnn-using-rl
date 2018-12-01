from networks import Cifar10VGG16
from agents import Agent
import gym
import numpy as np

if __name__ == '__main__':
    env = Cifar10VGG16()
    state = env.get_feature_map('block5_conv1')
    action_space = state.shape[0]
    state_size = state.shape[1]
    agent = Agent(state_size, action_space)
    print(agent.model.summary())

    state = np.reshape(state[1,:], [1, state_size])
    action = agent.get_action(state).astype(int)
    print(action)
    print('index')
    print(np.where(action == 0)[0])
