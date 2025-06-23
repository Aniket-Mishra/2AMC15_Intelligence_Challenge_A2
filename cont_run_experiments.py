import json
import subprocess
from copy import deepcopy
from itertools import product

# load the base config file
with open("agent_config.json", "r") as f:
    config = json.load(f)

# load the config file with all hyperparameters
with open("agent_config_multiple.json", "r") as f:
    all_params = json.load(f)

# non-model specific parameters
agents = ["dqn", "ppo"]
episodes = [1000, 2000]
steps = [5000]
learning_rates = [1e-3]
grids = ["table_easy", "table_hard"]

# for restoring the original config file, can be removed if not necessary
original_config = deepcopy(config)

# loop over every agent and hyperparameter combination
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
