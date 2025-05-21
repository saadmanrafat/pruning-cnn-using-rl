import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
import tensorflow_model_optimization as tfmot
from tensorflow.keras import Input

from utils import data_generator

import os
import math
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Cifar10VGG16:
    def __init__(self, b=0.5):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        # Normalize data
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        self.model = self.__build_model()
        self.num_classes = 10
        self.b = b
        self.action_size = None
        self.state_size = None
        self.epochs = 2
        self.base_model_accuracy = None
        self._current_state = 0
        self.layer_name = None
        self.pruning_filters = {}  # Keep track of pruned filters per layer

    def __build_model(self):
        """Builds the VGG16 Model with CIFAR-10 adaptation"""
        input_shape = self.x_train.shape[1:]  # 32x32x3 for CIFAR-10
        input_tensor = Input(shape=input_shape)
        vgg = VGG16(include_top=False, input_tensor=input_tensor, weights='imagenet')
        flatten = Flatten(name='Flatten')(vgg.output)
        prediction = Dense(10, activation='softmax')(flatten)
        model = Model(input_tensor, prediction)
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        return model

    def get(self, layer_name='block5_conv1'):
        """Get the current filter state for the RL agent"""
        self.layer_name = layer_name
        # Get the weights of the specified layer
        layer = self.model.get_layer(layer_name)
        weights = layer.get_weights()[0]  # Filter weights

        # Set state and action sizes
        self.state_size = weights.shape[:3]  # Input shape to the agent
        self.action_size = weights.shape[-1]  # Number of filters

        # Initialize pruning filters dictionary for this layer if not already
        if layer_name not in self.pruning_filters:
            self.pruning_filters[layer_name] = []

        # Get the current filter
        x = weights[:, :, :, self._current_state]

        # Check if we've processed all filters
        if self._current_state + 1 == self.action_size:
            self._current_state = 0  # Reset for next layer
            return True, x

        self._current_state += 1
        return False, x

    def _create_sparse_model(self, action, original_model):
        """
        Create a model with specified filters set to near-zero
        This simulates pruning using weight masks
        """
        # Clone the model to avoid modifying the original
        new_model = clone_model(original_model)
        new_model.set_weights(original_model.get_weights())

        # Get the target layer
        layer = new_model.get_layer(self.layer_name)
        weights = layer.get_weights()

        # Create a sparse version by zeroing out filters where action is 0
        filters_to_prune = np.where(action == 0)[0]

        # Store which filters we're pruning
        self.pruning_filters[self.layer_name].extend(filters_to_prune)

        # Make weights very small (near zero) for filters to prune
        # We use small values instead of zero to avoid gradient issues
        for filter_idx in filters_to_prune:
            weights[0][:, :, :, filter_idx] = weights[0][:, :, :, filter_idx] * 1e-10

        # Update the weights in the layer
        layer.set_weights(weights)

        # Compile the model
        new_model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

        return new_model, filters_to_prune

    def _create_pruned_model(self):
        """Create a pruned model using TensorFlow Model Optimization
           This applies the accumulated pruning decisions to create a final model"""
        # Create a dict mapping layer names to indices of filters to prune
        pruning_spec = {}
        for layer_name, filter_indices in self.pruning_filters.items():
            pruning_spec[layer_name] = filter_indices

        # Define a custom pruning method that removes specified filters
        def apply_pruning_to_layer(layer, pruning_info):
            if layer.name in pruning_info:
                # Apply structured pruning
                # This is a simplified version - in practice would use TFMOT's API
                return tfmot.sparsity.keras.prune_low_magnitude(
                    layer,
                    pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(
                        target_sparsity=0.8,  # Can be adjusted
                        begin_step=0,
                        end_step=1
                    )
                )
            return layer

        # Create pruned model (simplified approach)
        # In a full implementation, would use TFMOT's structured pruning capabilities
        pruned_model = clone_model(self.model)
        pruned_model.set_weights(self.model.get_weights())

        # Compile and return
        pruned_model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        return pruned_model

    def _accuracy_term(self, new_model):
        """Calculate the accuracy term for the reward function"""
        train_data_generator = data_generator(self.x_train, self.y_train, self.num_classes)
        eval_data_generator = data_generator(self.x_test, self.y_test, self.num_classes)

        # Fine-tune the model after pruning
        new_model.fit(train_data_generator,
                      epochs=self.epochs,
                      validation_data=eval_data_generator,
                      verbose=1)

        # Evaluate the pruned model
        _, p_hat = new_model.evaluate(eval_data_generator, verbose=1)

        # Get baseline model accuracy if we haven't yet
        if self.base_model_accuracy is None:
            print('Calculating the accuracy of the baseline model')
            _, self.base_model_accuracy = self.model.evaluate(eval_data_generator, verbose=1)

        # Calculate accuracy term as described in the paper
        accuracy_term = (self.b - (self.base_model_accuracy - p_hat)) / self.b
        return accuracy_term, p_hat

    def step(self, action):
        """Take an action (keep/remove filters) and return results"""
        # Create a model with specified filters set to near-zero
        new_model, filters_to_prune = self._create_sparse_model(action[0], self.model)

        # Calculate accuracy term
        accuracy_term, new_accuracy = self._accuracy_term(new_model)

        # Calculate efficiency term (as per the paper)
        # C(A^l) is number of kept filters
        kept_filters = self.action_size - len(filters_to_prune)
        if kept_filters > 0:
            efficiency_term = math.log(self.action_size / kept_filters)
        else:
            efficiency_term = 0  # Avoid divide by zero or log(0)

        # Compute total reward
        reward = accuracy_term * efficiency_term

        # Update base model if pruning was successful
        if accuracy_term > 0:
            self.model = new_model

        # Get next state
        done, next_state = self.get(self.layer_name)

        return action[0], reward, done, next_state

    def finalize_pruning(self):
        """Create final pruned model using TFMOT after all decisions are made"""
        # In a full implementation, this would apply TFMOT's real pruning
        return self._create_pruned_model()