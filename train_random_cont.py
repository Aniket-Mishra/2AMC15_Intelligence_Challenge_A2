import sys
import numpy as np
from agents.random_agent import RandomAgent
from world.cont_environment import Cont_Environment
from world.cont_path_visualizer import visualize_path_cont_env

from world.cont_grid import Grid

def main(grid_arg: str | None):
    """
    If grid_arg == "None", pass grid=None into Cont_Environment.
    Otherwise, we attempt to load the grid from cont_grid_config.py.
    """
    if grid_arg is None or grid_arg.lower() == "none":
        grid = None
        print("→ Running with no grid (grid=None).")
    else:
        try:
            from world.cont_grid_config import load_grid
        except ImportError:
            raise RuntimeError(
                "Could not find world/grid_config_cont.py or load_grid() inside it."
            )

        # load_grid() should return a 2D numpy array of ints
        raw_cells = load_grid()
        world_size = (4.0, 4.0)
        grid = Grid(raw_cells, world_size)
        print("→ Loaded grid from cont_grid_config.py.")

    env = Cont_Environment(
        no_gui=False,
        forward_speed=0.1,
        grid=grid,
        rotation_speed=np.pi / 12,
        agent_start_pos=(1.25, -1, 0.0),
        random_seed=42
    )

    agent = RandomAgent()
    agent_path = []

    try:
        state = env.reset()
        agent_path.append(state)
        done = False

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info, world_stats = env.step(action)
            state = next_state
            agent_path.append(state)

        print("Episode finished. Total steps:", info.get("total_steps", "N/A"))
        visualize_path_cont_env(env, agent_path)

    except KeyboardInterrupt:
        print("User closed the window. Exiting.")

    finally:
        if env.gui is not None:
            env.gui.close()


if __name__ == "__main__":
    #    python train_random_cont.py None       → grid=None
    #    python train_random_cont.py use_grid   → load from grid_config_cont.py
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
