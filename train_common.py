import argparse
import os
import numpy as np
import torch
import datetime
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
from agents.DQN_agent import DQNAgent
from agents.PPO_agent import PPOAgent
from agents.random_agent import RandomAgent
from world.cont_environment import Cont_Environment
from world.cont_grid import Grid
from world.cont_path_visualizer import visualize_path_cont_env
import math

AGENT_CLASSES = {
    "dqn": DQNAgent,
    "ppo": PPOAgent,
    "random": RandomAgent,
}

ROLLING_AVG = 50 # Nr of past episodes to take the rolling average reward over
CONVERGENCE_THRESHOLD = 0.95 # The agent is said to have converged when the rolling average of its last ROLLING_AVG rewards 
                             # first gets above CONVERGENCE_THRESHOLD * the highest reward earned over the course of training.

def get_agent_config(agent_name, state_dim, action_dim, args):
    '''Read in config file with agent parameters'''
    with open("agent_config.json", "r") as f:
        config = json.load(f)
    if agent_name == "dqn":
        parameters = config["DQN"]
        EPS_START = parameters["eps_start"]
        EPS_END = parameters["eps_end"]
        r = parameters["r"]
        ratio = parameters["ratio"]
        total_steps = args.episodes * args.max_steps
        t_target = int(ratio * total_steps)
        epsilon_decay = int(-t_target / math.log(r))
        return dict(
            state_dim=state_dim,
            action_dim=action_dim,
            epsilon_start=EPS_START,
            epsilon_end=EPS_END,
            epsilon_decay=epsilon_decay
        )
    elif agent_name == "ppo":
        parameters = config["PPO"]
        return dict(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=parameters["gamma"],
            clip_epsilon=parameters["clip_epsilon"],
            update_epochs=parameters["update_epochs"],
            batch_size=parameters["batch_size"],
            gae_lambda=parameters["gae_lambda"],
        )
    elif agent_name == "random":
        return {}
    else:
        raise ValueError(f"Unknown agent type: {agent_name}")


def get_filename(prefix, agent_name, agent_config, timestamp, extension):
    cfg_str = "_".join(f"{k}{v}" for k, v in agent_config.items())
    return f"{prefix}_{agent_name}_{cfg_str}_{timestamp}.{extension}"

# Wouter's new filename for use with plots_new.
# Once this has been fully integrated, clean up the old get_filename
def get_filename_new(folder, type, environment, agent, no_episodes, no_steps, lr, target_reward, extension):
    return f'{folder}/{type}_env={environment}_agent={agent}_episodes={no_episodes}_steps={no_steps}_lr={lr}_targetreward={target_reward}.{extension}'

def main(args):
    print("Provided args: ", args)
    args.lr = float(args.lr)
    # Initialize directories for saving log files
    Path("results_common/logs").mkdir(parents=True, exist_ok=True)
    Path("results_common/graph_path").mkdir(parents=True, exist_ok=True)
    Path("results_common/rewards").mkdir(parents=True, exist_ok=True)
    Path("results_common/graphs").mkdir(parents=True, exist_ok=True)

    ### STEP 1: INITIALIZE GRID ###
    if args.grid is None or args.grid == "none":
        grid = None
        starting_pos = (0, 0, 0.0)
        print("Running with no grid (grid=None).")
    elif args.grid == "wall":
        try:
            from world.wall_grid import load_grid
        except ImportError:
            raise RuntimeError(
                "Could not find world/wall_grid.py or load_grid() inside it."
            )
        # load_grid() should return a 2D numpy array of ints
        raw_cells, starting_pos = load_grid()
        world_size = (4.0, 4.0)
        grid = Grid(raw_cells, world_size, "wall_grid")
        print("Loaded grid from wall_grid.py.")
    elif args.grid == "table_easy":
        try:
            from world.table_grid_easy import load_grid
        except ImportError:
            raise RuntimeError(
                "Could not find world/table_grid_easy.py or load_grid() inside it."
            )
        # load_grid() should return a 2D numpy array of ints
        raw_cells, starting_pos = load_grid()
        world_size = (4.0, 4.0)
        grid = Grid(raw_cells, world_size, "table_grid_easy")
        print("Loaded grid from table_grid_easy.py.")
    elif args.grid == "table_hard":
        try:
            from world.table_grid_hard import load_grid
        except ImportError:
            raise RuntimeError(
                "Could not find world/table_grid_hard.py or load_grid() inside it."
            )
        # load_grid() should return a 2D numpy array of ints
        raw_cells, starting_pos = load_grid()
        world_size = (4.0, 4.0)
        grid = Grid(raw_cells, world_size, "table_grid_hard")
        print("Loaded grid from table_grid_hard.py.")
    else:
        raise ValueError(f"Unknown grid type: {args.grid}")

    env = Cont_Environment(no_gui=args.no_gui, 
                           grid=grid, 
                           random_seed=args.random_seed, 
                           agent_start_pos=starting_pos, 
                           target_reward=args.target_reward)
    state_dim = env.state_dim
    action_dim = env.action_dim

    ### STEP 2: SET UP AGENT ###
    agent_name = args.agent
    AgentClass = AGENT_CLASSES.get(agent_name)
    if AgentClass is None:
        raise ValueError(f"Unknown agent '{agent_name}'. Available: {list(AGENT_CLASSES)}")

    agent_cfg = get_agent_config(agent_name, state_dim, action_dim, args)
    if agent_cfg != {}:
        agent_cfg['lr'] = args.lr # Add lr from args
    print("Agent config: ", agent_cfg)
    agent = AgentClass(**agent_cfg) if agent_cfg else AgentClass()
    #print("TEST: ", agent.optimizer.param_groups[0]['lr'], env.target_reward) # See if lr and target reward were set correctly

    ### STEP 3: SET UP TRAINING PROCEDURE & TRACKERS ###
    episodes = args.episodes
    max_steps = args.max_steps

    metrics = []
    episode_rewards = []
    episode_lengths = []
    success_flags = []
    running_rewards = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #reward_file = get_filename("results_common/rewards/reward", agent_name, agent_cfg, timestamp, "csv") # OLD
    reward_file = get_filename_new("results_common/rewards", "METRICS", args.grid, args.agent, args.episodes, args.max_steps, args.lr, args.target_reward, 'csv') # NEW
    log_file = get_filename_new("results_common/logs", "LOG", args.grid, args.agent, args.episodes, args.max_steps, args.lr, args.target_reward, 'json') # NEW
    path_file = get_filename_new("results_common/graph_path", "GRAPH", args.grid, args.agent, args.episodes, args.max_steps, args.lr, args.target_reward, 'npy') # NEW
    visualization_file = get_filename_new("", "PATH", args.grid, args.agent, args.episodes, args.max_steps, args.lr, args.target_reward, 'png') # NEW
    visualization_file = visualization_file[1:] # get rid of / at the start

    update_every = 2 # PPO only learns every update_every episodes
    # Actually run training
    for ep in tqdm(range(1, episodes + 1)):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        success = 0

        # start a fresh per‐episode buffer
        if agent_name == "dqn":
            agent.start_episode()
        if agent_name == "ppo":
            agent.memory.clear()

        # Simulate 1 episode
        for t in range(1, max_steps + 1):
            if agent_name == "ppo":  # PPO
                action, logprob = agent.select_action(state)
                next_state, reward, done, info, _ = env.step(action)
                timeout = (t == max_steps) and (not done)
                agent.store_transition(state, action, logprob, reward, done)
            else:  # DQN/Random
                action = agent.take_action(state)
                next_state, reward, done, info, _ = env.step(action)
                timeout = (t == max_steps) and (not done)

                if agent_name == "dqn":
                    # record into the per‐episode buffer and learn off successful episodes
                    agent.record_transition(state, action, reward, next_state, done or timeout)
                    agent.learn()
                else:
                    if hasattr(agent, "store_transition") and agent_name != "random":
                        agent.store_transition(state, action, reward, next_state, done or timeout)
                    if hasattr(agent, "learn"):
                        agent.learn()

            state = next_state
            total_reward += reward
            steps = t
            if done or timeout:
                success = int(done)
                break

        if agent_name == "ppo":
            if ep % update_every == 0:
                agent.learn()

        # only for DQN: now that episode is done, push it into the main buffer if it succeeded
        if agent_name == "dqn":
            agent.end_episode(success)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_flags.append(success)
        running_rewards.append(np.mean(episode_rewards[-ROLLING_AVG:]))

        metrics.append({
            "episode": ep,
            "reward": total_reward,
            "length": steps,
            "success": success,
            f"avg_last_{ROLLING_AVG}": running_rewards[-1]
        })

        if(not args.no_print):
            print(f"Episode {ep:3d}: steps={steps}, total_reward={total_reward:.2f}, success={success}, avg_last_{ROLLING_AVG}={running_rewards[-1]:.2f}")

    if hasattr(agent, "finalize_training"):
        agent.finalize_training()

    # Calculate in which episode the agent converged
    avg_rewards = [ep_stats[f'avg_last_{ROLLING_AVG}'] for ep_stats in metrics][ROLLING_AVG:] # Ignore first episodes due to instability 
    max_reward = max(avg_rewards)
    rewards_converged = avg_rewards > CONVERGENCE_THRESHOLD * max_reward # True at index i iff the reward for episode i is > CONVERGENCE_THRESHOLD * max_reward
    convergence_episode = list(rewards_converged).index(True) if True in rewards_converged else len(rewards_converged) # First episode where the rewards converged
    convergence_episode += ROLLING_AVG # To account for the fact that we removed the first ROLLING_AVG episodes

    ### STEP 4: EVALUATE AGENT ###
    state = env.reset()
    path = [state]
    done = False
    with torch.no_grad():
        for t in range(1, max_steps + 1):
            action = agent.take_action(state)
            state, _, done, _, _ = env.step(action)
            path.append(state)
            if done:
                break

    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    np.save(path_file, np.array(path))
    print(f"Evaluation finished after {len(path) - 1} steps — path saved as {path_file}")

    # Save log of rewards earned
    os.makedirs(os.path.dirname(reward_file), exist_ok=True)
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(reward_file, index=False)
    print(f"Reward/metrics log saved as {reward_file}")

    # Make plot of rewards earned this run
    # NOTE: No longer used, we now use plots_new.py which plots multiple runs in a single figure
    reward_png = reward_file.replace("/rewards/", "/graphs/").replace(".csv", ".png")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df_metrics['reward'], mode='lines', name='Reward per Episode'))
    fig.add_trace(go.Scatter(y=df_metrics[f'avg_last_{ROLLING_AVG}'], mode='lines', name=f'Avg Last {ROLLING_AVG}'))
    fig.update_layout(
        title=f"Episode Rewards - {agent_name.upper()}",
        xaxis_title="Episode",
        yaxis_title="Reward",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        template="plotly_white",
        width=900, height=400
    )
    # Unused
    #fig.write_image(reward_png)
    #print(f"Saved Plotly reward curve as: {reward_png}")

    # Save image visualizing the path the agent took
    path_png = visualize_path_cont_env(env, path, filename=visualization_file)
    print(f"Saved PIL path visualization as: {path_png}")

    # Save log of config used for this run and where the saved logs/images can be found
    log_dict = {
        "agent": agent_name,
        "agent_config": agent_cfg,
        "timestamp": timestamp,
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": args.random_seed,
        "success_rate": float(np.mean(success_flags)),
        f"avg_reward_last_{ROLLING_AVG}": float(np.mean(episode_rewards[-ROLLING_AVG:])),
        "final_eval_steps": len(path) - 1,
        "reward_file": reward_file,
        "path_file": path_file,
        "reward_plot_png": reward_png,
        "path_plot_png": path_png,
        "EVALUATION STEPS": len(path),
        "CONVERGENCE EPISODE": convergence_episode
    }
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(log_dict, f, indent=2)
    print(f"Full training log saved as {log_file}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a random agent in a continuous environment with optional grid and GUI"
    )
    p.add_argument(
        "--grid",
        choices=["none", "wall", "table_easy", "table_hard"],
        default="none",
        help="Which grid to load: none, wall, or table (default: none).",
    )
    p.add_argument(
        "--agent",
        choices=["random", "dqn", "ppo"],
        default="dqn",
        help="Select agent to train: random, DQN, ppo (default: dqn).",
    )

    p.add_argument(
        "--episodes",
        type=int,
        action="store",
        default=1000,
        help="Define number of episodes to train (default: 2000).",
    )

    p.add_argument(
        "--max-steps",
        type=int,
        action="store",
        default=500,
        help="Define maximum number of steps to train (default: 10000).",
    )

    p.add_argument(
        "--random-seed",
        action="store",
        default=42,
        help="Define random seed (default: 42).",
    )

    p.add_argument(
        "--target-reward",
        action="store",
        default=300,
        help="Reward received for reaching the target."
    )

    p.add_argument(
        "--lr",
        action="store",
        default=1e-3,
        help="Learning rate for the agent."
    )

    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without GUI (default: False)."
    )

    p.add_argument(
        "--no-print",
        action="store_true",
        help="Run without printing the summary of each episode (default: False)."
    )

    return p.parse_args()


"""
ARGUMENTS
--grid [none, wall, table_easy, table_hard]
--agent [random, DQN, ppo]
--episodes Integer
--max-steps Integer
--target-reward Integer
--random-seed Integer
--lr Integer
--no-gui
--no-print
"""
if __name__ == "__main__":
    args = parse_args()
    main(args)
