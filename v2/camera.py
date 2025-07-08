# camera.py
import pygame

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

class Camera:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.target_x = 0
        self.target_y = 0
        self.zoom = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 2.0
        self.follow_speed = 0.1

    def update(self, target_x, target_y):
        self.target_x = target_x
        self.target_y = target_y
        self.x = self.x + (self.target_x - self.x) * self.follow_speed
        self.y = self.y + (self.target_y - self.y) * self.follow_speed

    def adjust_zoom(self, delta):
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom + delta))

    def apply_transform(self, surface, pos):
        screen_x = (pos[0] - self.x) * self.zoom + WINDOW_WIDTH / 2
        screen_y = (pos[1] - self.y) * self.zoom + WINDOW_HEIGHT / 2
        return screen_x, screen_y

    def apply_surface_transform(self, surface, pos):
        scaled_surface = pygame.transform.scale(
            surface,
            (int(surface.get_width() * self.zoom), int(surface.get_height() * self.zoom))
        )
        return scaled_surface

    def getpos(self, pos):
        screen_x = (pos[0] - self.x) * self.zoom + WINDOW_WIDTH / 2
        screen_y = (pos[1] - self.y) * self.zoom + WINDOW_HEIGHT / 2
        return screen_x, screen_y
