"""
Environment.
"""
import random
import datetime
import numpy as np
from tqdm import trange
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime
from world.helpers import save_results, action_to_direction
from world.cont_path_visualizer import visualize_path_cont_env

try:
    from agents import BaseAgent
    from world.cont_gui import GUI
    from world.cont_grid import Grid
    from world.path_visualizer import visualize_path
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.append(root_path)

    from agents import BaseAgent
    from world.gui import GUI
    from world.path_visualizer import visualize_path



class Cont_Environment:
    def __init__(self,
                 forward_speed: float = 0.2,
                 rotation_speed: float = np.pi/6,
                 grid: Grid = None,
                 no_gui: bool = False,
                 #sigma: float = 0.,
                 agent_start_pos: tuple[float, float, float] = None, # Start pos is defined by (x,y,phi) coordinates
                 reward_fn: callable = None,
                 #target_fps: int = 30,
                 random_seed: int | float | str | bytes | bytearray | None = 0):
        
        """Creates the Continuous Environment for the Reinforcement Learning robot.
        For now, the implementation is very basic; the world is empty save for the agent and the boundaries.

        OLD DESCRIPTION:
        This environment follows the general principles of reinforcment
        learning. It can be thought of as a function E : action -> observation
        where E is the environment represented as a function.

        Args:
            grid_fp: Path to the grid file to use. <- UNUSED; once we have obstacles figured out, replace with a path to the obstacle file (whatever format that may be in)
            no_gui: True if no GUI is desired.
            sigma: The stochasticity of the environment. The probability that
                the agent makes the move that it has provided as an action is
                calculated as 1-sigma. <- UNUSED
            agent_start_pos: Tuple where each agent should start.
                If None is provided, then a random start position is used.
            reward_fn: Custom reward function to use.  <- UNUSED until target is implemented
            target_fps: How fast the simulation should run if it is being shown
                in a GUI. If in no_gui mode, then the simulation will run as fast as
                possible. We may set a low FPS so we can actually see what's
                happening. Set to 0 or less to unlock FPS. <- UNUSED until GUI is done
            random_seed: The random seed to use for this environment. If None
                is provided, then the seed will be set to 0.
        """
        self.grid = grid
        self.forward_speed = forward_speed
        self.rotation_speed = rotation_speed
        self.agent_start_pos = agent_start_pos
        self.terminal_state = False
        self.state_dim  = 3   # because reset() returns (x,y,phi) # If you change the number of features also change this
        self.action_dim = 3   # because step() accepts 0,1,2,3 only # If you change the number of actions also change this

        # Initialize environment if a grid was provided
        if grid is not None:
            if grid.get_name() == "wall_grid":
                print("Wall grid successfuly loaded")
                self.grid = grid
                # Override x_bounds, y_bounds to match the world_size used by the Grid:
                self.x_bounds = [grid.x_min, grid.x_min + grid.world_width]
                self.y_bounds = [grid.y_min, grid.y_min + grid.world_height]

                self.target_pos = np.array([1.25, 1.25])
                self.target_radius = self.forward_speed * 1.5

            elif grid.get_name() == "table_grid":
                print("Table grid successfuly loaded")
                self.grid = grid
                # Override x_bounds, y_bounds to match the world_size used by the Grid:
                self.x_bounds = [grid.x_min, grid.x_min + grid.world_width]
                self.y_bounds = [grid.y_min, grid.y_min + grid.world_height]

                self.target_pos = np.array([0.75, 0.75])
                self.target_radius = self.forward_speed * 2

            else:
                raise ValueError("Grid name not found")

        # Otherwise if None, create a 4x4 world and create a target at a uniformly random place within the grid
        else:
            self.x_bounds = [-2,2]
            self.y_bounds = [-2,2]

            # Target and target radius
            self.target_pos = np.array([random.uniform(self.x_bounds[0], self.x_bounds[1]),
                                        random.uniform(self.y_bounds[0], self.y_bounds[1])])
            self.target_radius = self.forward_speed * 2    # Size of the target. 1.5 times the step size so that it cannot overshoot
              
        # Set up reward function
        if reward_fn is None:
            warn("No reward function provided. Using default reward.")
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn

        # GUI specific code: Set up the environment as a blank state.
        self.no_gui = no_gui
        if not self.no_gui:
            world_width = self.x_bounds[1] - self.x_bounds[0]
            world_height = self.y_bounds[1] - self.y_bounds[0]
            self.gui = GUI(world_size=(world_width, world_height),
                           window_size=(1152, 768),
                           fps=30)
            self.gui.reset()
        else:
            self.gui = None

    def _reset_info(self) -> dict:
        """Resets the info dictionary.

        info is a dict with information of the most recent step
        consisting of whether the target was reached or the agent
        moved and the updated agent position.
        """
        return {"target_reached": False,
                "agent_moved": False,
                "actual_action": None}
    
    @staticmethod
    def _reset_world_stats() -> dict:
        """Resets the world stats dictionary.

        world_stats is a dict with information about the 
        environment since last env.reset(). Basically, it
        accumulates information.
        """
        return {"cumulative_reward": 0,
                "total_steps": 0,
                "total_agent_moves": 0,
                "total_failed_moves": 0,
                "total_targets_reached": 0,
                }

    def _initialize_agent_pos(self):
        """Initializes agent position and rotation from the given location.
        If none is provided, places the agent at (0,0) with a random rotation.
        """

        if self.agent_start_pos is not None:
            pos = (self.agent_start_pos[0], self.agent_start_pos[1], self.agent_start_pos[2])
            # TODO: Check if this position is not inside an obstacle or inside target region

            # Check if position is not out of bounds
            if self.out_of_bounds(pos):
                raise ValueError("Attempted to place agent out of bounds!")
            
            # Otherwise, the position is safe; place agent there
            self.agent_pos = pos
        else:
            # No position given. We place agent at (0,0).
            warn("No initial agent positions provided. Placing agent at (0,0) with a random rotation.")
            # TODO: make this a random location not inside an obstacle?
            self.agent_pos = (0,0, np.random.uniform(0, 2*np.pi))


    def reset(self, **kwargs) -> tuple[float, float, float]:
        """Reset the environment to an initial state.

        You can fit it keyword arguments which will overwrite the 
        initial arguments provided when initializing the environment.

        Args:
            **kwargs: possible keyword options are the same as those for
                the environment initializer.
        Returns:
             initial state.
        """
        for k, v in kwargs.items():
            # Go through each possible keyword argument.
            match k:
                case "grid_fp":
                    self.grid_fp = v
                case "agent_start_pos":
                    self.agent_start_pos = v
                case "no_gui":
                    self.no_gui = v
                case "target_fps":
                    self.target_spf = 1. / v
                case _:
                    raise ValueError(f"{k} is not one of the possible "
                                     f"keyword arguments.")
        
        # Reset variables
        #self.grid = Grid.load_grid(self.grid_fp).cells # TODO: reload environment layout file
        self._initialize_agent_pos()
        self.terminal_state = False
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()

        if self.gui is not None:
            self.gui.reset()

        return self.agent_pos

    def _move_agent(self, new_pos: tuple[float, float, float]):
        """Moves the agent if possible and updates the corresponding stats.

        Args:
            new_pos: The new (x,y,phi) position of the agent.
        """
        # Check if the new position is out of bounds (can maybe be combined with check above)
        if self.out_of_bounds(new_pos):
            self.world_stats["total_failed_moves"] += 1
            self.info["agent_moved"] = False
            pass
        # Check if the new position is in the target region
        # TODO
        # Otherwise, moved to an empty tile
        else:
            self.agent_pos = new_pos
            self.info["agent_moved"] = True
            self.world_stats["total_agent_moves"] += 1
        

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """This function makes the agent take a step in the world.

        Action is provided as integer and values are:
            - 0: Move forward
            - 1: Rotate left
            - 2: Rotate right
            - 3: Stop (do nothing)
        Args:
            action: Integer representing the action the agent should
                take. 

        Returns:
            0) Current state,
            1) The reward for the agent,
            2) If the terminal state has been reached (currently unused since there is no target)
        """

        # GUI specific code
        if self.gui is not None:
            #If we are paused AND the user has NOT pressed the right-arrow or next step button, just redraw
            if self.gui.paused and not self.gui.step_requested:
                paused_info = self._reset_info()
                paused_info["agent_moved"] = False
                self.gui.render(self.agent_pos, paused_info, self.world_stats, 0.0, self.grid, False, self.target_pos, self.target_radius)
                return self.agent_pos, 0.0, self.terminal_state, self.info, self.world_stats

            #If single-step was requested, clear that flag and proceed exactly one step
            if self.gui.step_requested:
                is_single_step = True
                self.gui.step_requested = False
            else:
                is_single_step = False
        else:
            # GUI is disabled so no pause/single-step logic
            is_single_step = False

        self.world_stats["total_steps"] += 1

        # Add stochasticity into the agent action
        # TODO: Implement later? (not sure if we even want this at all)
        #val = random.random()
        #if val > self.sigma:
        #    actual_action = action
        #else:
        #    actual_action = random.randint(0, 3)
        actual_action = action
        
        # Make the move
        self.info["actual_action"] = actual_action

        if actual_action == 0: # Move forward in direction specified by phi
            new_pos = (self.agent_pos[0] + self.forward_speed*np.cos(self.agent_pos[2]), # x + fwd_speed*cos(phi)
                       self.agent_pos[1] + self.forward_speed*np.sin(self.agent_pos[2]), # y + fwd_speed*sin(phi)
                       self.agent_pos[2])                                                # phi (unchanged)
        elif actual_action == 1: # Rotate left by rotation_speed radians
            new_pos = (self.agent_pos[0], self.agent_pos[1], (self.agent_pos[2] + self.rotation_speed) % (2*np.pi))
        elif actual_action == 2: # Rotate right by rotation_speed radians
            new_pos = (self.agent_pos[0], self.agent_pos[1], (self.agent_pos[2] - self.rotation_speed) % (2*np.pi))
        else: # Stop (do nothing) - useless for now, but this will be replaced with a "decelerate" action later.
            new_pos = self.agent_pos

        # out‐of‐bounds: stay, big penalty, terminate
        if self.out_of_bounds(new_pos):
            reward = -10 #SET LOW
            self.terminal_state = False
            self.world_stats["total_failed_moves"] += 1
            self.info["agent_moved"] = False
            if self.gui is not None:
                self.gui.render(
                    self.agent_pos,
                    self.info,
                    self.world_stats,
                    reward,
                    self.grid,
                    is_single_step,
                    self.target_pos,
                    self.target_radius
                )
            return self.agent_pos, reward, self.terminal_state, self.info, self.world_stats

        # Obstacle collision check
        if self.grid is not None:
            if self.grid.is_obstacle(new_pos[0], new_pos[1]):
                # Apply the same penalty as “out-of-bounds”
                reward = -10

                self.world_stats["total_failed_moves"] += 1
                self.info["agent_moved"] = False

                if self.gui is not None:
                    self.gui.render(
                        self.agent_pos,
                        self.info,
                        self.world_stats,
                        reward,
                        self.grid,
                        is_single_step,
                        self.target_pos,
                        self.target_radius
                    )

                # Early‐return exactly as out-of-bounds would
                return self.agent_pos, reward, self.terminal_state, self.info, self.world_stats

        # Calculate the reward for the agent
        # TODO; for now reward is always -1
        reward = self.reward_fn(self, new_pos)

        self._move_agent(new_pos)
        
        self.world_stats["cumulative_reward"] += reward

        # GUI specific code
        if self.gui is not None:
            self.gui.render(self.agent_pos, self.info, self.world_stats, reward, self.grid, is_single_step, self.target_pos, self.target_radius)

        return self.agent_pos, reward, self.terminal_state, self.info, self.world_stats

    @staticmethod
    def _default_reward_function(self, new_pos) -> float:
        """Simple reward function for initial testing.
        The reward should be:
        - Slightly negative for taking a safe step (to encourage finding the shortest path)
        - Strongly negative for running into an obstacle
        - Very strongly positive for reaching the target region.

        Currently target and obstacles are not implemented, so the reward is an arbitrary -0.1 for taking a safe step
        and -10 for running into a wall (values subject ot change later).

        Args:
            new_pos: The position the agent is moving to.
            Add more arguments here as needed (once target and obstacles are implemented)

        Returns:
            A single floating point value representing the reward for a given action.
        """
        if self.out_of_bounds(new_pos):
            reward = -10
        elif np.linalg.norm(np.array(new_pos[:2]) - self.target_pos) <= self.target_radius:
            reward = 10
            self.terminal_state = True
            self.info["target_reached"] = True
        else:
            reward = -0.1

        return reward
    
    def out_of_bounds(self, pos) -> bool:
        '''Helper function that checks if given position is out of bounds'''
        return not(self.x_bounds[0] < pos[0] < self.x_bounds[1] and self.y_bounds[0] < pos[1] < self.y_bounds[1])

    @staticmethod
    def evaluate_agent(#grid_fp: Path,
                       agent: BaseAgent,
                       max_steps: int,
                       #sigma: float = 0.,
                       agent_start_pos: tuple[int, int] = None,
                       random_seed: int | float | str | bytes | bytearray = 0,
                       show_images: bool = False):
        """Evaluates a single trained agent's performance.

        Currently all this does is run the agent, it does not save or visualize any output.
        
        OLD DESCRIPTION:
        What this does is it creates a completely new environment from the
        provided grid and does a number of steps _without_ processing rewards
        for the agent. This means that the agent doesn't learn here and simply
        provides actions for any provided observation.

        For each evaluation run, this produces a statistics file in the out
        directory which is a txt. This txt contains the values:
        [ 'total_steps`, `total_failed_moves`]

        Args:
            grid_fp: Path to the grid file to use. <- UNUSED until obstacles are done
            agent: Trained agent to evaluate.
            max_steps: Max number of steps to take. 
            sigma: same as abve. <- UNUSED
            agent_start_pos: same as above.
            random_seed: same as above.
            show_images: Whether to show the images at the end of the
                evaluation. If False, only saves the images. <- UNUSED until GUI is done
        """

        env = Cont_Environment(#grid_fp=grid_fp,
                          no_gui=True,
                          #sigma=sigma,
                          agent_start_pos=agent_start_pos,
                          #target_fps=-1,
                          random_seed=random_seed)
        
        state = env.reset()
        #initial_grid = np.copy(env.grid)

        # Add initial agent position to the path
        agent_path = [env.agent_pos]

        for _ in trange(max_steps, desc="Evaluating agent"):
            
            action = agent.take_action(state)
            state, _, terminated, _, _ = env.step(action)

            agent_path.append(state)

            if terminated:
                break
        
        # Once evaluation is done (early‐stop or full run), save the path image:
        visualize_path_cont_env(env, agent_path)
        print(f"\n✅ Evaluation finished after {len(agent_path)-1} steps — trajectory saved in results/") 
        #env.world_stats["targets_remaining"] = np.sum(env.grid == 3)

        # TODO: Replace visualize_path function
        #path_image = visualize_path(initial_grid, agent_path)
        #file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

        #save_results(file_name, env.world_stats, path_image, show_images)