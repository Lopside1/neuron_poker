"""
Player based on a trained neural network using Ray RLlib’s new API stack 
with RLModule (PyTorch). This code defines a custom DQN RLModule with three 
dense layers and dropout, sets up an RLlib DQN agent using the new API stack, 
and provides training and evaluation (play) methods.
"""

import time
import json
import logging
import gym
import numpy as np

import tune

import ray
from ray import tune
from ray import air
from ray.rllib.algorithms.dqn import DQN, DQNConfig  # New API: use ray.rllib.algorithms

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the TorchRLModule base class from the new RLModule API.
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

# Set up logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Define a custom RLModule for DQN.
#
# We subclass TorchRLModule and implement the three required methods:
#   - forward_train: for computing Q-values during training (with dropout active)
#   - forward_inference: for evaluation (without dropout)
#   - forward_exploration: for action selection during exploration (can use dropout noise)
#
# The network architecture is similar to your original model.
# -----------------------------------------------------------------------------
class CustomDQNModule(TorchRLModule):
    def __init__(self, observation_space, action_space, num_outputs, model_config, name, **kwargs):
        # Call parent constructors.
        TorchRLModule.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        
        # Determine input size from the observation space.
        input_size = int(np.product(observation_space.shape))
        
        # Build a network with three Dense layers (512 units each) with dropout.
        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 512)
        self.dropout3 = nn.Dropout(0.2)
        self.fc_out = nn.Linear(512, num_outputs)
    
    def forward_train(self, input_dict, **kwargs):
        # input_dict is expected to contain the observation under key "obs"
        x = input_dict["obs"]
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # dropout active during training
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        q = self.fc_out(x)
        return {"q_values": q}

    def forward_inference(self, input_dict, **kwargs):
        # In inference mode, disable dropout.
        self.eval()  # switch to evaluation mode
        with torch.no_grad():
            x = input_dict["obs"]
            if len(x.shape) > 2:
                x = torch.flatten(x, start_dim=1)
            x = F.relu(self.fc1(x))
            # Note: dropout layers are effectively bypassed in eval mode.
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            q = self.fc_out(x)
        self.train()  # switch back to train mode for future calls
        return {"q_values": q}

    def forward_exploration(self, input_dict, **kwargs):
        # For exploration, we can simply use the training forward pass.
        # (If you wish to incorporate noise differently during exploration,
        #  adjust this method accordingly.)
        return self.forward_train(input_dict, **kwargs)


# -----------------------------------------------------------------------------
# Define the Player class using RLlib’s new DQN API with RLModule (PyTorch).
#
# Notice: instead of using ModelCatalog and the "custom_model" key, we now use
# the new .rl_module() builder on the config to set our custom RLModule.
# -----------------------------------------------------------------------------
class Player:
    def __init__(self, name="DQN", env_name=None, config_overrides=None, checkpoint_path=None):
        """
        :param name: Name for the agent.
        :param env_name: Gym environment id (e.g., "CartPole-v1" or your custom env).
        :param config_overrides: (Optional) dict to override default RLlib DQN config.
        :param checkpoint_path: (Optional) path to a checkpoint to restore from.
        """
        self.name = name
        self.env_name = env_name
        self.checkpoint_path = checkpoint_path

        self.config = (
            #TODO For a complete rainbow setup, make the following changes to the default DQN config: "n_step": [between 1 and 10], "noisy": True, "num_atoms": [more than 1], "v_min": -10.0, "v_max": 10.0 (set v_min and v_max according to your expected range of returns).
            DQNConfig()
            .environment(env=env_name)
            .training(
                num_atoms=tune.grid_search([1,])
            )
        )

        # Restore from checkpoint if provided.
        if checkpoint_path:
            raise NotImplementedError("Restoring from checkpoint is not implemented yet. Code needs to be updated.")
            self.agent.restore(checkpoint_path)
            log.info(f"Restored agent from checkpoint: {checkpoint_path}")

    def train(self, num_iterations=100):
        self.tuner = tune.Tuner(
            "DQN",
            run_config=air.RunConfig(
                stop={"training_iteration": num_iterations}
                ),
            param_space=self.config
        )
        self.tuner.fit()

    def play(self, num_episodes=5, render=False):
        """Run the trained agent in the environment for a number of episodes."""
        env = gym.make(self.env_name)
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                if render:
                    env.render()
                # Use the compute_single_action method.
                action = self.agent.compute_single_action(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            log.info(f"Episode {episode}: total reward = {total_reward}")
        env.close()

    def get_action(self, obs):
        """Return the action for a given observation."""
        return self.agent.compute_single_action(obs)


# -----------------------------------------------------------------------------
# Example usage: training and evaluation.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize Ray.
    ray.init(ignore_reinit_error=True)

    # Use a Gym environment (replace with your custom env if needed).
    env_name = "CartPole-v1"
    player = Player(name="DQN", env_name=env_name)

    # Train the agent for 100 iterations (adjust as needed).
    player.train(num_iterations=100)

    # Evaluate the trained agent for 5 episodes (set render=True to visualize).
    player.play(num_episodes=5, render=False)

    # Shutdown Ray when finished.
    ray.shutdown()
