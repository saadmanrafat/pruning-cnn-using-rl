from environments import Cifar10VGG16
from agents import Agent
import numpy as np
import os
import tensorflow as tf

# Enable memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Create directory for saved models
os.makedirs('./saved_model', exist_ok=True)

if __name__ == '__main__':
    # Layers to prune in sequence (from lower to higher as per paper)
    layers_to_prune = [
        'block1_conv1', 'block1_conv2',
        'block2_conv1', 'block2_conv2',
        'block3_conv1', 'block3_conv2', 'block3_conv3',
        'block4_conv1', 'block4_conv2', 'block4_conv3',
        'block5_conv1', 'block5_conv2', 'block5_conv3'
    ]

    # Main environment for pruning
    env = Cifar10VGG16(b=1.0)  # Adjust b to control performance-pruning tradeoff

    # Number of episodes to train for each layer
    episodes = 3

    # Prune each layer sequentially
    for layer_name in layers_to_prune:
        print(f"\n===== Pruning layer: {layer_name} =====")

        # Train agent for this layer
        for episode in range(episodes):
            print(f"Episode {episode + 1}/{episodes}")

            # Reset environment state for this layer
            done, state = env.get(layer_name=layer_name)

            # Create or load agent
            agent = Agent(env.state_size, env.action_size)

            # Process all filters in the layer
            while not done:
                # Get action probabilities
                action_probs = agent.get_action(state)

                # Convert to binary actions (1: keep, 0: prune)
                # Use stochastic sampling for exploration
                action = np.random.binomial(1, action_probs)

                # Take step in environment
                action_taken, reward, done, new_state = env.step(action)

                # Store experience for training
                agent.append_sample(state, action_taken, reward)
                print(f'Filter {env._current_state - 1}: Reward {reward}')

                # Update state
                state = new_state

            # Train agent after processing all filters
            agent.train_model()

            # Save agent weights
            agent.model.save_weights(f'./saved_model/pruning_agent_{layer_name}.h5')

    # Create final pruned model using TFMOT
    print("\n===== Creating final pruned model =====")
    final_pruned_model = env.finalize_pruning()

    # Save the pruned model
    final_pruned_model.save('./saved_model/pruned_vgg16.h5')

    print("Pruning completed successfully!")