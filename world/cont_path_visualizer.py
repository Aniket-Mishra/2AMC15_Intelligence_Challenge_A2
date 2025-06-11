from PIL import Image, ImageDraw
import numpy as np
import os
from datetime import datetime

def visualize_path_cont_env(env, agent_path, save_dir="results", image_size=None):
    """Creates and saves an image of the path the agent took.

    Args:
        env: the Cont_Environment instance.
        agent_path: a list of position tuples, one per time step.
        save_dir: where to save the image.
        image_size: size of the output image.

    Returns:
        nothing
    """

    # Create a new directory if not there yet and create filename
    os.makedirs(save_dir, exist_ok=True)
    if image_size is None:
        if getattr(env, "gui", None) is not None:
            # we have a GUI—use its world‐rect dimensions
            image_size = (env.gui.world_rect.width, env.gui.world_rect.height)
        else:
            # no GUI—pick a reasonable default canvas size
            image_size = (902, 768)

    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    save_path = os.path.join(save_dir, f"{timestamp}.png")

    width, height = image_size
    img = Image.new("RGB", image_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    def world_to_pixel(pos):
        x, y = pos
        x_norm = (x - env.x_bounds[0]) / (env.x_bounds[1] - env.x_bounds[0])
        y_norm = (env.y_bounds[1] - y) / (env.y_bounds[1] - env.y_bounds[0])
        return int(x_norm * width), int(y_norm * height)

    # Draw path
    path_points = [world_to_pixel((x, y)) for x, y, _ in agent_path]
    draw.line(path_points, fill=(0, 0, 255), width=2)

    # Draw target
    target_px = world_to_pixel(env.target_pos)
    target_radius_world = env.target_radius
    px_radius = int(target_radius_world * width / (env.x_bounds[1] - env.x_bounds[0]))
    draw.ellipse([target_px[0] - px_radius, target_px[1] - px_radius,
                  target_px[0] + px_radius, target_px[1] + px_radius],
                 fill=(0, 200, 0), outline=None)

    # Save image
    img.save(save_path)
