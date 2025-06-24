import json
import subprocess
from copy import deepcopy
from itertools import product

# load the base config file
#with open("agent_config.json", "r") as f:
#    config = json.load(f)

# load the config file with all hyperparameters
#with open("agent_config_multiple.json", "r") as f:
#    all_params = json.load(f)

# non-model specific parameters
agents = ["ppo"]
grids = ["table_easy", "table_hard"]

#params_to_test = ['Episodes', 'Steps', 'Learning rate', 'Target reward']
params_to_test = ['Episodes', 'Learning rate', 'Target reward']
EPISODES = [500, 1000, 2000] # Default value is tested here, doesn't need to be tested for the other ones
STEPS = [250, 500, 1000]
LEARNING_RATES = [1e-2, 1e-3, 1e-4]
TARGET_REWARDS = [100, 300, 700]

# for restoring the original config file, can be removed if not necessary
#original_config = deepcopy(config)

def get_test_values(param_to_test):
    '''Returns the values to test and corresponding command line argument depending on parameter to test.'''
    match param_to_test:
        case 'Episodes':
            return EPISODES, '--episodes'
        case 'Steps':
            return STEPS, '--max-steps'
        case 'Learning rate':
            return LEARNING_RATES, '--lr'
        case 'Target reward':
            return TARGET_REWARDS, '--target-reward'

# loop over every agent and hyperparameter combination
for (param_to_test, grid, agent) in product(params_to_test, grids, agents):
    print((param_to_test, grid, agent))

for (param_to_test, grid, agent) in product(params_to_test, grids, agents):
    # Get param values
    test_values, cmd_argument = get_test_values(param_to_test)

    for test_value in test_values:
        # Build cmd command. Non-test values are kept at default.
        cmd = [
            "python", "train_common.py",
            "--no-gui", "--no-print",
            "--agent", agent,
            "--grid", grid,
            cmd_argument, str(test_value),
        ]
        print(f"\n === NOW RUNNING Grid = {grid} | Agent = {agent} | {param_to_test} = {test_value} ===")
        subprocess.run(cmd)


''' 
# Natasha's version. 
# I (Wouter) opted to put all the variable parameters in cmd arguments instead so we don't need to mess with the config files.
for agent, params in all_params.items():
    keys, values = zip(*params.items())
    
    for combination in product(*values):
        param_combo = dict(zip(keys, combination))

        for episode, step, grid in product(episodes, steps, grids):
            # create new config for this iteration
            agent_key = agent.upper()
            config_copy = deepcopy(config)
            config_copy[agent_key].update(param_combo)

            # overwrite current agent_config
            with open("agent_config.json", "w") as f:
                json.dump(config_copy, f, indent=2)

            # cmd command
            cmd = [
                "python", "train_common.py",
                "--agent", agent,
                "--grid", grid,
                "--episodes", str(episode),
                "--max-steps", str(step),
                "--no-gui"
            ]

            print(f"\n[running] {agent} | {param_combo} | episodes={episode}, max_steps={step}, grid={grid}")
            subprocess.run(cmd)

# for restoring the original config file, can be removed if not necessary
with open("agent_config.json", "w") as f:
    json.dump(original_config, f, indent=2)
'''