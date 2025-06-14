# world/cont_gui.py

import pygame
from world.cont_grid import Grid
import numpy as np
from pygame.locals import QUIT, KEYDOWN, K_SPACE, K_RIGHT, MOUSEBUTTONDOWN


class GUI:
    """
    Pygame GUI for a continuous (x, y, φ) environment.
    """

    def __init__(self,
                 world_size: tuple[float, float],
                 window_size: tuple[int, int] = (1152, 768),
                 fps: int = 30):
        pygame.init()
        pygame.display.init()

        # Basic parameters
        self.world_width, self.world_height = world_size
        self.screen_width, self.screen_height = window_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Continuous Environment")
        self.clock = pygame.time.Clock()
        self.fps = fps

        # Pause / single-step state
        self.paused = False
        self.step_requested = False

        # Stats (updated every render by reading `info`)
        self.cumulative_reward = 0.0
        self.total_steps = 0

        # Preload a font for rendering text
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)

        # Colors
        self.bg_color = (250, 250, 250)
        self.obstacle_color = (100, 100, 100)
        self.target_color = (0, 255, 0)
        self.agent_color = (0, 0, 255)
        self.arrow_color = (255, 0, 0)
        self.text_color = (0, 0, 0)
        self.panel_bg = (230, 230, 230)
        self.divider_color = (0, 0, 0)
        self.panel_width = 250

        # Split the window: left for world, right for info
        self.world_rect = pygame.Rect(0, 0,
                                      self.screen_width - self.panel_width,
                                      self.screen_height)
        self.info_rect = pygame.Rect(self.world_rect.width, 0,
                                     self.panel_width,
                                     self.screen_height)

        # These will be updated each frame
        self.pause_button_rect = None
        self.step_button_rect = None

        self.last_render_time = pygame.time.get_ticks()

    def world_to_screen(self, pos: tuple[float, float, float]) -> tuple[int, int]:
        """
        Map a world coordinate (x, y, φ) into pixel (sx, sy).
        We ignore φ for positioning; φ is used only to draw the orientation arrow.
        """
        x, y, _ = pos
        norm_x = (x + (self.world_width / 2)) / self.world_width
        norm_y = ((self.world_height / 2) - y) / self.world_height

        sx = int(self.world_rect.left + norm_x * self.world_rect.width)
        sy = int(self.world_rect.top + norm_y * self.world_rect.height)
        return sx, sy

    def reset(self):
        """
        Called at the start of each episode. Clears stats and redraws an empty frame.
        """
        self.cumulative_reward = 0.0
        self.total_steps = 0
        self.screen.fill(self.bg_color)
        pygame.display.flip()

    def render(self,
               agent_pos: tuple[float, float, float],
               info: dict,
               world_stats: dict,
               reward: float,
               grid: Grid = None,
               is_single_step: bool = False,
               target_pos: tuple[float, float] = None,
               target_radius: float = None):
        """
        Draw the current frame. Call once per environment step.

        Args:
            grid: grid the agent should run on (None for now)
            agent_pos: (x, y, φ)
            info: contains other info on agent (not sure whether this is relevant here)
            world_stats: contains total steps
            reward: float reward from this step
            is_single_step: True if this call comes from a single-step action
            target_pos: coordinates of the target
            target_radius: size of the target
        """

        # Update stats from world_stats + reward
        self.total_steps = world_stats.get("total_steps", self.total_steps)
        self.cumulative_reward += reward

        # Clear the world area
        self.screen.fill(self.bg_color, self.world_rect)

        # Draw the target (circular)
        if target_pos is not None and target_radius is not None:
            target_screen_pos = self.world_to_screen((target_pos[0], target_pos[1], 0))
            scaled_radius = int(target_radius * (self.world_rect.width / self.world_width))
            pygame.draw.circle(self.screen, (0, 200, 0), target_screen_pos, scaled_radius)

        # Draw each obstacle cell as a filled rect before drawing agent/target:
        if grid is not None:
            for (r, c) in grid.get_all_obstacle_cells():
                x0 = grid.x_min + c * grid.cell_width
                y0 = grid.y_min + r * grid.cell_height
                x1 = x0 + grid.cell_width
                y1 = y0 + grid.cell_height

                screen_x0, screen_y0 = self.world_to_screen((x0, y0, 0.0))
                screen_x1, screen_y1 = self.world_to_screen((x1, y1, 0.0))

                left = min(screen_x0, screen_x1)
                top = min(screen_y0, screen_y1)
                width = abs(screen_x1 - screen_x0)
                height = abs(screen_y1 - screen_y0)

                pygame.draw.rect(self.screen,
                                 self.obstacle_color,
                                 pygame.Rect(left, top, width, height))

        # Draw the agent (circle + arrow)
        sx, sy = self.world_to_screen(agent_pos)
        pygame.draw.circle(self.screen, self.agent_color, (sx, sy), 8)
        arrow_length = 20
        phi = agent_pos[2]
        ex = sx + int(arrow_length * np.cos(phi))
        ey = sy - int(arrow_length * np.sin(phi))
        pygame.draw.line(self.screen, self.arrow_color, (sx, sy), (ex, ey), 2)

        # Draw divider
        div_x = self.world_rect.width
        pygame.draw.line(self.screen, self.divider_color,
                         (div_x, 0), (div_x, self.screen_height), 3)

        # Draw the info panel
        pygame.draw.rect(self.screen, self.panel_bg, self.info_rect)

        # Draw stats text
        lines = [
            f"Total Steps: {self.total_steps}",
            f"Cumulative Reward: {self.cumulative_reward:.2f}",
            f"Agent X: {agent_pos[0]:.2f}",
            f"Agent Y: {agent_pos[1]:.2f}",
            f"Agent Rotation: {agent_pos[2]:.2f}",
        ]
        text_x = self.info_rect.left + 10
        text_y = 10
        line_height = 30
        for idx, line in enumerate(lines):
            txt_surf = self.font.render(line, True, self.text_color)
            self.screen.blit(txt_surf, (text_x, text_y + idx * line_height))

        # Draw Pause/Resume button
        button_w, button_h = 150, 40
        pause_y = text_y + len(lines) * line_height + 30
        pause_rect = pygame.Rect(text_x, pause_y, button_w, button_h)
        self.pause_button_rect = pause_rect

        pause_color = (200, 200, 200) if self.paused else (255, 255, 255)
        pygame.draw.rect(self.screen, pause_color, pause_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), pause_rect, 1)
        label = "Resume" if self.paused else "Pause"
        txt = self.font.render(label, True, self.text_color)
        txt_pos = txt.get_rect(center=pause_rect.center)
        self.screen.blit(txt, txt_pos)

        # Draw “Take Single Step” button
        step_y = pause_y + button_h + 20
        step_rect = pygame.Rect(text_x, step_y, button_w, button_h)
        self.step_button_rect = step_rect

        # If not paused, render in gray and ignore clicks
        if self.paused:
            step_color = (255, 255, 255)
        else:
            step_color = (180, 180, 180)  # grayed out when not paused
        pygame.draw.rect(self.screen, step_color, step_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), step_rect, 1)
        txt2 = self.font.render("Take Single Step", True, self.text_color)
        txt2_pos = txt2.get_rect(center=step_rect.center)
        self.screen.blit(txt2, txt2_pos)

        # Flip buffers & cap FPS
        pygame.display.flip()
        self.clock.tick(self.fps)

        # Event loop (after drawing so buttons show up immediately)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                raise KeyboardInterrupt("User closed window")

            # Spacebar toggles pause
            elif event.type == KEYDOWN and event.key == K_SPACE:
                self.paused = not self.paused
                if not self.paused:
                    self.step_requested = False

            # If paused and user presses right-arrow, set single-step
            elif event.type == KEYDOWN and event.key == K_RIGHT and self.paused:
                self.step_requested = True

            # Mouse clicks: check if buttons were clicked?
            elif event.type == MOUSEBUTTONDOWN:
                mx, my = event.pos

                # Pause/Resume button clicked?
                if pause_rect.collidepoint(mx, my):
                    self.paused = not self.paused
                    if not self.paused:
                        self.step_requested = False

                # Take Single Step button clicked?
                elif step_rect.collidepoint(mx, my) and self.paused:
                    self.step_requested = True

        pygame.event.pump()

    def close(self):
        pygame.quit()
