# 2AMC15 Data Intelligence Challenge 2025 - Group 10
The following README contains instructions on how to replicate the results found in the associated report of group 10, as well as a short description of the algorithms and environment used for this reinforcement learning task.

## Theoretical overview
### Environment
The problem we focus on is that of a robot navigating a restaurant to deliver food to customers. The robot is capable of three actions: *Move forward*, *Rotate right*, and *Rotate left*. To simulate a restaurant environment, obstacles in the form of tables (shown as gray boxes) have been added to the 2D plane representing the restaurant. The robot's objective is to reach the target, which is represented as a green circle. 

Two obstacle layouts ("grids") are provided, one easy (`table_grid_easy`) and one more difficult (`table_grid_hard`). Besides these, there is also a `wall_grid` which models the restaurant as a whole instead of just the dining area, with walls being the main obstacle rather than tables. This grid was ultimately cut from testing, but is left in the files as a starting point for future research.

This repository implements two RL algorithms for training the robot, **DQN** (Deep Q-Network) and **PPO** (Proximal Policy Optimization). Besides this, there is also an agent which makes random moves, which was used as a dummy for testing.

### DQN
DQN extends the tabular Q-learning algorithm from Assignment 1 to our continuous state space. 
Rather than storing a lookup table of Q-values, it approximates the action-value function using a neural network, implemented as a 3-layer MLP with two hidden layers of dimension 128. It takes the 3 state features as input, and outputs one scalar Q-value for each of the 3 actions. A variety of techniques are also implemented to speed up the learning process; details on these can be found in the report.

### PPO
Proximal Policy Optimization (PPO) is a policy gradient method that optimizes a stochastic policy directly, providing stability and efficiency through a clipped objective. PPO is suitable for both discrete and continuous action spaces, making it ideal for navigation tasks with unknown or changing goals.
PPO improves classic policy gradients by constraining the policy update with a clipped objective to avoid large, destabilizing updates.

## Files
`../agents` - Contains all agent files; `DQN_agent.py` contains the logic for the DQN agent and `PPO_agent.py` for the PPO agent.

`../world` - contains all files related to the environment. `cont_environment.py` contains logic for step taking of agents, collision detection, and GUI updates. `table_grid_easy.py`, `table_grid_hard.py` and `wall_grid.py` contain configurations for the respective obstacle layouts and can be modified/mimicked to create your own environments.

`../results`, `../results_common` - stores the results including graphs as found in the report.

`train_common.py` - Script to train a single agent on a specified grid with provided arguments.

`cont_run_experiments.py` - Runs the experiments as outlined in the report.

`agent_config.json` - Stores the configuration of agents; modify parameters here before training the agents.

## Usage
### Training individual agents
Using the `train_common.py` file, it is possible to train individual agents using specific settings. The parameters of the desired agent can be modified in the `agent_config.json` file. Only certain parameters are found in this file; other parameters are passed as arguments in the command line, as listed below:
```
--grid [none, wall, table_easy, table_hard] - "Which grid to load: none, wall, table_easy or table_hard (default: table_easy)."
--agent [random, DQN, ppo] - "Which agent to train: random, dqn, or ppo (default: dqn)."
--episodes (Integer) - "Number of episodes to train for (default: 1000)."
--max-steps (Integer) - "(Max) number of steps per episode (default: 500)."
--random-seed (Integer) - "Random seed to use the same initialization for different agents (default: 42)."
--target-reward (Integer) - "Reward received for reaching the target."
--lr (Integer) - "Learning rate of the agent."
--no-gui - "Run without GUI (default: False)."
--no-print - "Run without printing the summary of each episode (default: False)."
```
Example runs

`python train_common.py --agent dqn --grid table_easy` - Runs the DQN agent on the easy table grid with the default test settings (1000 episodes, 500 steps/episode, etc.)

`python train_common.py --agent ppo --grid table_hard --episodes 500 --no-gui` - Runs the PPO agent on the hard table grid, for just 500 episodes and without showing the GUI.

### Running the experiments found in the report
Running the `cont_run_experiments.py` file as shown below performs the same experiments as shown in the report. This file runs a suite of separate experiments with differing parameters and grids. WARNING: it can take several hours to run all experiments!

Command: `python cont_run_experiments.py`