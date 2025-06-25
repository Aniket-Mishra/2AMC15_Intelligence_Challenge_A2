# Data Intelligence Challenge 2025 - Group 10
The following information contains instructions on how to replicate the results found in the associated report of group 10, as well as a short description of the algorithms and environment used for this reinforcement learning task.
## Environment and Agents
Our RL task consists of a robot navigating a restaurant to deliver food to customers. The robot is capable of three actions: Move forward, Rotate right, Rotate left. To simulate a restaurant environment, obstacles in the form of tables have been added to the grid representing the restaurant. There are two possible grids, one easy (`table_grid_easy`) and one more difficult (`table_grid_hard`). There are two possible RL algorithms to choose from for the robot, DQN (Deep Q-Network) or PPO (Proximal Policy Optimization), with the possibility to also initialize a random agent which makes random moves.
### DQN
DQN extends the tabular Q-learning algorithm from Assignment 1 to continuous state
spaces and discrete action sets.
Rather than storing a lookup table of Q-values, we approximate the action-value function
by a neural network, implemented as a 3-layer MLP of which 2 layers have 128 neurons
each. It takes the 3 state features and outputs one scalar Q-value per action.
### PPO
Proximal Policy Optimization (PPO) is a policy gradient method that optimizes a stochastic policy directly, providing stability and efficiency through a clipped objective. PPO is
suitable for both discrete and continuous action spaces, making it ideal for navigation tasks
with unknown or changing goals.
PPO improves classic policy gradients by constraining the policy update with a clipped
objective to avoid large, destabilizing updates.
## Files
`../agents` - contains all agent files

`../world` - contains all files related to the environment. `cont_environment.py` contains logic for step taking of agents, collision detection, GUI updates.

`../results`, `../results_common` - stores the results including graphs as found in the report

`train_DQN.py`, `train_ppo.py`, `train_random_cont.py` - legacy training files, obsolete as of latest implementation

`train_common.py` - Script to train a single agent on a specified grid with provided arguments

`cont_run_experiments.py` - Runs the predefined experiments as outlined in the report

`agent_config.json`, `agent_config_multiple.json` - stores the configuration of agents, with the latter file storing multiple values to run multiple agents
## Usage
### Training individual agents
Using the `train_common.py` file it is possible to train individual agents. To change the parameters of the desired agent use the `agent_config.json` file. Only certain parameters are found in this file, other parameters that are passed as arguments in the command line are listed below:
```
--grid [none, wall, table_easy, table_hard] - "Which grid to load: none, wall, or table (default: none)."
--agent [random, DQN, ppo] - "Select agent to train: random, DQN, ppo (default: dqn)."
--episodes (Integer) - "Define number of episodes to train (default: 2000)."
--max-steps (Integer) - "Define maximum number of steps to train (default: 10000)."
--target-reward (Integer) - "Reward received for reaching the target."
--random-seed (Integer) - "Define random seed (default: 42)."
--lr (Integer) - "Learning rate for the agent."
--no-gui - "Run without GUI (default: False)."
--no-print - "Run without printing the summary of each episode (default: False)."
```
Example runs

`python train_common.py --agent dqn --grid table_easy`

`python train_common.py --agent ppo --grid table_hard --episodes 500 --no-gui`

### Running the experiments found in the report
By running the `cont_run_experiments.py` file, it is possible to get the same results as shown in the report. This file runs 6 separate experiments with differing parameters and grids.

`python cont_run_experiments.py`