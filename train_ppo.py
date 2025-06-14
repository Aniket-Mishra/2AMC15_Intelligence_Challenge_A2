import sys
import numpy as np
import torch

from agents.PPO_agent import PPOAgent
from world.cont_environment import Cont_Environment
from world.cont_path_visualizer import visualize_path_cont_env
from world.wall_grid import load_grid
from world.cont_grid import Grid
import math

def main(grid_arg, episodes, max_steps, random_seed):
    # Load or omit grid
    if grid_arg is None or grid_arg.lower() == "none":
        grid = None
        print(" Running with no grid (4×4 world)")
    else:
        raw_cells = load_grid()
        grid = Grid(raw_cells, world_size=(4.0, 4.0))
        print(" Loaded grid for 4×4 world.")

    # Create environment
    env = Cont_Environment(no_gui=True, grid=grid, random_seed=random_seed)

    # Derive state & action dimensions from the environment
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = PPOAgent(
        state_dim,
        action_dim,
        gamma=0.99,
        lr=3e-4,
        clip_epsilon=0.1,
        update_epochs=10,
        batch_size=64,
        gae_lambda=0.95
    )

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        for t in range(1, max_steps + 1):
            action, logprob = agent.select_action(state)
            next_state, reward, done, info, _ = env.step(action)
            timeout = (t == max_steps) and (not done)
            agent.store_transition(state, action, logprob, reward, done or timeout)
            state = next_state
            total_reward += reward
            if done or timeout:
                break

        agent.learn()

        print(f"Episode {ep:3d}: steps={t}, total_reward={total_reward:.2f}")

    agent.finalize_training()

    # **Single** evaluation run
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


    visualize_path_cont_env(env, path)
    print(f"\n Evaluation finished after {len(path)-1} steps — trajectory saved in results/")

if __name__ == "__main__":
    grid_arg = sys.argv[1] if len(sys.argv) > 1 else None
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    random_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    main(
        grid_arg,
        episodes,
        max_steps,
        random_seed
    )

# call python3 train_PPO.py to train