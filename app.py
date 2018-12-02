from networks import Cifar10VGG16
from agents import Agent
import numpy as np

if __name__ == '__main__':

    while True:

        env = Cifar10VGG16('block5_conv1')
        state = env.get_feature_map()
        state = np.reshape(state[1,:], [1, env.state_size])
        agent = Agent(env.state_size, env.action_size)

        for episode in range(5):
            action = agent.get_action(state).astype(int)
            action, reward = env.step(action)
            agent.append_sample(state, action, reward)

        agent.train_model()
        agent.model.save_weights('./save_model/pruning_agent.h5')
