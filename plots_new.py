# Script to plot the *average reward over last N episodes* for one experiment.
# Produces one figure per experiment (i.e. parameter being varied) and per grid, with 6 lines;
# 3 solid ones in different colors for DQN with different settings, and 3 dashed ones in those same colors for PPO with different settings.

import os
import pandas
import matplotlib.pyplot as plt
from itertools import product

DATA_DIR = 'results_common/rewards' # Folder where the metrics per episode are stored
LOG_DIR = 'results_common/logs'     # Folder where the metrics for the whole run are stored
OUTPUT_DIR = 'results_common/plots' # Folder to output plots to
os.makedirs(OUTPUT_DIR, exist_ok=True)

### VALUES TO PLOT ###
# Note: plot only one environment at a time, since the optimal path is different between grids.
ENV_TO_PLOT = 'table_easy'      # Which environment to make a plot for, 'Easy' or 'Hard'
PARAM_TO_PLOT = 'Learning rate' # Which parameter to make a plot for, 'Episodes', 'Steps', 'Learning rate' or 'Target reward'

ROLLING_AVG = 50          # Make sure this is the same as the one the experiments were carried out with
COLORS = ['r', 'g', 'b']
AGENTS = ['dqn', 'ppo']

# Experimental values; default value should always be the middle one
EPISODES_VALUES = [500, 1000, 2000] 
STEPS_VALUES = [250, 500, 1000]
LR_VALUES = [1e-2, 1e-3, 1e-4]
TARGET_REWARD_VALUES = [100, 300, 700]
######################

### AUXILIARY FUNCTIONS ###
def read_data_file(environment, agent, no_episodes, no_steps, lr, target_reward):
    '''Reads in a data csv and returns the episodes and corresponding rolling average rewards to use as x and y for plotting.
    Also returns the episode nr where the reward has converged.'''
    param_string = f'env={environment}_agent={agent}_episodes={no_episodes}_steps={no_steps}_lr={lr}_targetreward={target_reward}.csv'
    full_data = pandas.read_csv(f'{DATA_DIR}/METRICS_{param_string}', header=0)
    #log_data = pandas.read_json(f'{LOG_DIR}/LOG_{param_string}')
    return full_data['episode'], full_data[f'avg_last_{ROLLING_AVG}']#, log_data['CONVERGENCE EPISODE']

def get_label(param_to_plot, agent, no_episodes, no_steps, lr, target_reward):
    '''Returns the correct label for a line depending on which parameter needs to be plotted'''
    label = f'{agent}'.upper() + ' | '
    match param_to_plot:
        case 'Episodes':
            label += f'{no_episodes} episodes'
        case 'Steps':
            label += f'{no_steps} steps'
        case 'Learning rate':
            label += f'lr = {lr}'
        case 'Target reward':
            label += f'reward = {target_reward}'
    return label

###########################

### ACTUALLY MAKE THE PLOT ###
# Note: this is based on the parameters provided in the VALUES TO PLOT section above, make sure to set them correctly there!
def make_plot(env_to_plot, param_to_plot):
    # Select only default values for all parameters except the one we want to plot.
    # Also initialize plot title & such
    episodes = [EPISODES_VALUES[1]]
    steps = [STEPS_VALUES[1]]
    lrs = [LR_VALUES[1]]
    target_rewards = [TARGET_REWARD_VALUES[1]]

    match param_to_plot:
        case 'Episodes':
            episodes = EPISODES_VALUES
        case 'Steps':
            steps = STEPS_VALUES
        case 'Learning rate':
            lrs = LR_VALUES
        case 'Target reward':
            target_rewards = TARGET_REWARD_VALUES

    # Plot the 6 lines as mentioned at the top
    for agent in AGENTS:
        color_id = 0
        linestyle = '--' if agent == 'ppo' else '-'

        for (no_episodes, no_steps, lr, target_reward) in product(episodes, steps, lrs, target_rewards):
            x, y = read_data_file(env_to_plot, agent, no_episodes, no_steps, lr, target_reward)
            label = get_label(param_to_plot, agent, no_episodes, no_steps, lr, target_reward)
            plt.plot(x, y, color=COLORS[color_id], linestyle=linestyle, label=label)
            #plt.plot(convergence_episode, 0, color=COLORS[color_id], marker='o') # Plot a point on the x-axis where the convergence episode is reached
            color_id += 1

    plt.xlabel('Episode')
    plt.ylabel('Avg reward')
    plt.title(f'Avg of last {ROLLING_AVG} rewards on {env_to_plot} grid, varying {param_to_plot}')
    plt.grid()
    plt.legend(loc='lower right')
    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/plot_env={env_to_plot}_parameter={param_to_plot}'
    plt.savefig(save_path)
    
    print(f"Succesfully saved plot to {save_path}")

if __name__ == "__main__":
    make_plot(ENV_TO_PLOT, PARAM_TO_PLOT)