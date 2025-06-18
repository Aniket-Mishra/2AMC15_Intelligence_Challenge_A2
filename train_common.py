import argparse
import sys
import os
import numpy as np
import torch
import datetime
import json
from pathlib import Path

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


#TODO I removed most of the values for DQN and PPO from args, because thats too many args to pass. We can add these to a config file instead
def get_agent_config(agent_name, state_dim, action_dim, args):
    if agent_name == "dqn":
        EPS_START = 1.0
        EPS_END = 0.01
        r = 0.01
        ratio = 0.3
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
        return dict(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=0.99,
            lr=3e-4,
            clip_epsilon=0.1,
            update_epochs=10,
            batch_size=64,
            gae_lambda=0.95
        )
    elif agent_name == "random":
        return {}
    else:
        raise ValueError(f"Unknown agent type: {agent_name}")


def get_filename(prefix, agent_name, agent_config, timestamp, extension):
    cfg_str = "_".join(f"{k}{v}" for k, v in agent_config.items())
    return f"{prefix}_{agent_name}_{cfg_str}_{timestamp}.{extension}"


def main(args):
    print(args)
    Path("results_common/logs").mkdir(parents=True, exist_ok=True)
    Path("results_common/graph_path").mkdir(parents=True, exist_ok=True)
    Path("results_common/rewards").mkdir(parents=True, exist_ok=True)
    Path("results_common/graphs").mkdir(parents=True, exist_ok=True)

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
    else:
        raise ValueError(f"Unknown grid type: {args.grid}")

    env = Cont_Environment(no_gui=args.no_gui, grid=grid, random_seed=args.random_seed, agent_start_pos=starting_pos)
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent_name = args.agent
    AgentClass = AGENT_CLASSES.get(agent_name)
    if AgentClass is None:
        raise ValueError(f"Unknown agent '{agent_name}'. Available: {list(AGENT_CLASSES)}")

    agent_cfg = get_agent_config(agent_name, state_dim, action_dim, args)
    agent = AgentClass(**agent_cfg) if agent_cfg else AgentClass()

    episodes = args.episodes
    max_steps = args.max_steps

    metrics = []
    episode_rewards = []
    episode_lengths = []
    success_flags = []
    running_rewards = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reward_file = get_filename("results_common/rewards/reward", agent_name, agent_cfg, timestamp, "csv")
    log_file = get_filename("results_common/logs/log", agent_name, agent_cfg, timestamp, "json")
    path_file = get_filename("results_common/graph_path/path", agent_name, agent_cfg, timestamp, "npy")

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        success = 0

        if agent_name == "ppo":
            agent.memory.clear()

        for t in range(1, max_steps + 1):
            if agent_name == "ppo":
                action, logprob = agent.select_action(state)
                next_state, reward, done, info, _ = env.step(action)
                timeout = (t == max_steps) and (not done)
                agent.store_transition(state, action, logprob, reward, done or timeout)
            else:
                action = agent.take_action(state)
                next_state, reward, done, info, _ = env.step(action)
                timeout = (t == max_steps) and (not done)
                if hasattr(agent, "store_transition"):
                    if agent_name == "random":
                        pass
                    else:
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
            agent.learn()

        if agent_name == "dqn" and hasattr(agent, "target_update") and ep % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_flags.append(success)
        running_rewards.append(np.mean(episode_rewards[-100:]))

        metrics.append({
            "episode": ep,
            "reward": total_reward,
            "length": steps,
            "success": success,
            "avg_last_100": running_rewards[-1]
        })

        print(
            f"Episode {ep:3d}: steps={steps}, total_reward={total_reward:.2f}, success={success}, avg_last100={running_rewards[-1]:.2f}")

    if hasattr(agent, "finalize_training"):
        agent.finalize_training()

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

    os.makedirs(os.path.dirname(reward_file), exist_ok=True)
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(reward_file, index=False)
    print(f"Reward/metrics log saved as {reward_file}")

    reward_png = reward_file.replace("/rewards/", "/graphs/").replace(".csv", ".png")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df_metrics['reward'], mode='lines', name='Reward per Episode'))
    fig.add_trace(go.Scatter(y=df_metrics['avg_last_100'], mode='lines', name='Avg Last 100'))
    fig.update_layout(
        title=f"Episode Rewards - {agent_name.upper()}",
        xaxis_title="Episode",
        yaxis_title="Reward",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        template="plotly_white",
        width=900, height=400
    )
    fig.write_image(reward_png)
    print(f"Saved Plotly reward curve as: {reward_png}")

    path_png = visualize_path_cont_env(env, path)
    print(f"Saved PIL path visualization as: {path_png}")

    log_dict = {
        "agent": agent_name,
        "agent_config": agent_cfg,
        "timestamp": timestamp,
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": args.random_seed,
        "success_rate": float(np.mean(success_flags)),
        "avg_reward_last100": float(np.mean(episode_rewards[-100:])),
        "final_eval_steps": len(path) - 1,
        "reward_file": reward_file,
        "path_file": path_file,
        "reward_plot_png": reward_png,
        "path_plot_png": path_png
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
        choices=["none", "wall", "table_easy"],
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
        help="Define number of episodes to train (default: 1000).",
    )

    p.add_argument(
        "--max-steps",
        type=int,
        action="store",
        default=100,
        help="Define maximum number of steps to train (default: 100).",
    )

    p.add_argument(
        "--random-seed",
        action="store",
        default=42,
        help="Define random seed (default: 42).",
    )

    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Run with or without GUI (default: True).",
    )

    return p.parse_args()


"""
ARGUMENTS
--grid [none, wall, table_easy]
--agent [random, DQN, ppo]
--episodes Integer
--max-steps Integer
--random-seed Integer
--no-gui
"""
if __name__ == "__main__":
    args = parse_args()
    main(args)
