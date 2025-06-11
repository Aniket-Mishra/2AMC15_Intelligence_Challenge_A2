import sys
import numpy as np

from agents.DQN_agent import DQNAgent
from world.cont_environment import Cont_Environment
from world.cont_path_visualizer import visualize_path_cont_env
from world.cont_grid_config import load_grid
from world.cont_grid import Grid
import math


def main(grid_arg, episodes, max_steps, random_seed):
    # Load or omit grid
    if grid_arg is None or grid_arg.lower() == "none":
        grid = None
        print("→ Running with no grid (4×4 world)")
    else:
        raw_cells = load_grid()
        grid = Grid(raw_cells, world_size=(4.0, 4.0))
        print("→ Loaded grid for 4×4 world.")

    # Create environment
    env = Cont_Environment(no_gui=True, grid=grid, random_seed=random_seed)

    # Derive state & action dimensions from the environment
    state_dim = env.state_dim
    action_dim = env.action_dim

    EPS_START = 1.0
    EPS_END   = 0.01
    r         = 0.01            # residual fraction above ε_end at t_target
    ratio     = 0.3             # 80% into training

    total_steps = episodes * max_steps          # 2_500_000
    t_target    = int(ratio * total_steps)      # 2_000_000

    epsilon_decay = int(-t_target / math.log(r))  # ≈ 434_294

    agent = DQNAgent(
    state_dim,
    action_dim,
    epsilon_start=EPS_START,
    epsilon_end=EPS_END,
    epsilon_decay=epsilon_decay
    )

    # Training loop
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        for t in range(1, max_steps + 1):
            action = agent.take_action(state)
            next_state, reward, done, info, _ = env.step(action)
            timeout = (t == max_steps) and (not done)
            agent.store_transition(state, action, reward, next_state, done or timeout)
            agent.learn()
            state = next_state
            total_reward += reward
            if done or timeout:
                break

        # Periodically update target network
        if ep % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(f"Episode {ep:3d}: steps={t}, total_reward={total_reward:.2f}")

    # Switch to greedy evaluation policy
    agent.finalize_training()

    # **Single** evaluation run
    state = env.reset()
    path = [state]
    done = False
    for t in range(1, max_steps + 1):
        action = agent.take_action(state)
        state, _, done, _, _ = env.step(action)
        path.append(state)
        if done:
            break

    visualize_path_cont_env(env, path)
    print(f"\n✅ Evaluation finished after {len(path)-1} steps — trajectory saved in results/")

if __name__ == "__main__":
    # Positional args via sys.argv: grid_arg, episodes, max_steps, random_seed
    grid_arg = sys.argv[1] if len(sys.argv) > 1 else None
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    random_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    main(
        grid_arg,
        episodes,
        max_steps,
        random_seed
    )

 # call python3 train_dqn.py to train