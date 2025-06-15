import sys
import numpy as np
import math

from agents.DQN_agent import DQNAgent
from world.cont_environment import Cont_Environment
from world.cont_path_visualizer import visualize_path_cont_env
from world.cont_grid_config import load_grid
from world.cont_grid import Grid

def reward_function(reward, state, next_state, done):
    step_penalty = -0.1
    rotation_penalty = abs(state[2] - next_state[2]) * -0.01
    if done and reward > 0:
        return 100.0
    elif done:
        return -50.0
    return reward + step_penalty + rotation_penalty

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

    # Derive dimensions from the environment
    state_dim = env.state_dim
    action_dim = env.action_dim

    # Epsilon schedule
    EPS_START = 1.0
    EPS_END = 0.01
    epsilon_decay = 50_000
    failure_weight = 1.0


    agent = DQNAgent(
        state_dim,
        action_dim,
        epsilon_start=EPS_START,
        epsilon_end=EPS_END,
        epsilon_decay=epsilon_decay,
        failure_weight=failure_weight
    )


    state = env.reset()
    for _ in range(1000):
        action = np.random.randint(action_dim)
        next_state, reward, done, _, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()

    # Training loop
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        for t in range(1, max_steps + 1):
            action = agent.take_action(state)
            next_state, reward, done, info, _ = env.step(action)
            timeout = (t == max_steps) and (not done)

            reward = reward_function(reward, state, next_state, done)

            agent.store_transition(state, action, reward, next_state, done or timeout)
            agent.learn()
            agent.learn()

            state = next_state
            total_reward += reward

            if done or timeout:
                break

        # periodically update target network
        if ep % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        eps = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * math.exp(
            -agent.steps_done / agent.epsilon_decay)
        print(f"Episode {ep:3d}: steps={t}, total_reward={total_reward:.2f}, ε={eps:.4f}")

    # Evaluation (greedy)
    agent.finalize_training()
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
    print(f"\n Evaluation finished after {len(path) - 1} steps — trajectory saved in results/")

if __name__ == "__main__":
    # Args via sys.argv
    grid_arg = sys.argv[1] if len(sys.argv) > 1 else None
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    random_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    main(grid_arg, episodes, max_steps, random_seed)

# call python3 train_dqn.py to train