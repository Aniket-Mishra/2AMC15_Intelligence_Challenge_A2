import numpy as np
from agents.random_agent import RandomAgent
from world.cont_environment import Cont_Environment
from world.cont_path_visualizer import visualize_path_cont_env

def main():
    """
    Temporary train file that we can decide to extend if it is easier then reimplementing the old ones
    """
    env = Cont_Environment(
        no_gui=False,
        forward_speed=0.1,
        rotation_speed=np.pi / 12,
        agent_start_pos=(0.0, 0.0, 0.0),
        random_seed=42
    )

    agent = RandomAgent()
    agent_path = []    # for the agent path

    try:
        state = env.reset()
        agent_path.append(state)
        done = False

        while not done:
            action = agent.take_action(state)          # returns 0,1,2,3 with your probabilities
            next_state, reward, done, info = env.step(action)
            state = next_state
            agent_path.append(state)

        print("Episode finished. Total steps:", info.get("total_steps", "N/A"))

        # Save the image of the agent path
        visualize_path_cont_env(env, agent_path)

    except KeyboardInterrupt:
        print("User closed the window. Exiting.")

    finally:
        if env.gui is not None:
            env.gui.close()

if __name__ == "__main__":
    main()