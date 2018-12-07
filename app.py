from environments import Cifar10VGG16
from agents import Agent
import numpy as np

if __name__ == '__main__':
    for i in range(5):
        env = Cifar10VGG16()
        done, state = env.get()
        agent = Agent(env.state_size, env.action_size)
        while not done:
            action = agent.get_action(state)
            action = np.where(action > 0.5, 1, 0)
            action, reward, done, new_state = env.step(action)
            agent.append_sample(state, action, reward)
            print('State {}: Reward {}'.format(env._current_state - 1, reward))
            state = new_state
            if done:
                agent.train_model()
        agent.model.save_weights('./saved_model/pruning_agent.h5')