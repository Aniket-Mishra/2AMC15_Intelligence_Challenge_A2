import numpy as np
from agents.random_agent import RandomAgent
from world.cont_environment import Cont_Environment
from world.cont_path_visualizer import visualize_path_cont_env
import argparse

from world.cont_grid import Grid

def main(grid_arg: str| None, no_gui: bool):
    """
    If grid_arg == "None", pass grid=None into Cont_Environment.
    Otherwise, we attempt to load the grid from either from wall_grid.py or table_grid_easy.py.
    """
    if grid_arg is None or grid_arg.lower() == "none":
        grid = None
        starting_pos = (0, 0, 0.0)
        print("→ Running with no grid (grid=None).")
    elif grid_arg.lower() == "wall":
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
        print("→ Loaded grid from wall_grid.py.")
    elif grid_arg.lower() == "table_easy":
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
        print("→ Loaded grid from table_grid_easy.py.")

    env = Cont_Environment(
        no_gui=no_gui,
        forward_speed=0.2,
        grid=grid,
        rotation_speed=np.pi / 6,
        agent_start_pos=starting_pos,
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

        print("Episode finished. Total steps:", world_stats.get("total_steps", "N/A"))
        visualize_path_cont_env(env, agent_path)

    except KeyboardInterrupt:
        print("User closed the window. Exiting.")

    finally:
        if env.gui is not None:
            env.gui.close()

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a random agent in a continuous environment with optional grid and GUI"
    )
    p.add_argument(
        "--grid",
        choices=["none", "wall", "table"],
        default="none",
        help="Which grid to load: none, wall, or table (default: none).",
    )
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without the GUI even if a grid is loaded.",
    )

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # convert "none" to None
    grid_arg = None if args.grid.lower() == "none" else args.grid
    main(
        grid_arg=grid_arg,
        no_gui=args.no_gui,
    )