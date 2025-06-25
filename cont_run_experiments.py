import json
import subprocess
from copy import deepcopy
from itertools import product

### PARAMETERS TO TEST - MODIFY HERE ###
agents = ["dqn", "ppo"]
grids = ["table_easy", "table_hard"]
params_to_test = ['Episodes', 'Steps', 'Learning rate', 'Target reward']

EPISODES = [500, 1000, 2000] # Default values are tested here, doesn't need to be tested for the other ones
STEPS = [250, 1000]
LEARNING_RATES = [1e-2, 1e-4]
TARGET_REWARDS = [100, 700]

### AUXILIARY FUNCTIONS ###
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

# Loop over every agent and hyperparameter combination
# First do a test print to let the user know which values are about to be tested
print("=== TESTING THE FOLLOWING COMBINATIONS ===")
for (param_to_test, grid, agent) in product(params_to_test, grids, agents):
    print((param_to_test, grid, agent))

# Now actually run the experiments
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