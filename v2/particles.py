import pygame
import random
import math
import time as tms  # Import time with alias tms

class Explosion:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.start_time = tms.time()
        self.lifetime = 0.5

    def update(self, delta_time):
        elapsed = tms.time() - self.start_time
        return elapsed < self.lifetime

    def draw(self, screen, camera):
        if not self.update(0):
            return
        elapsed = tms.time() - self.start_time
        alpha = max(0, 255 * (1 - elapsed / self.lifetime))
        size = 40 * (elapsed / self.lifetime)
        surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (255, 100, 0, int(alpha)), (size, size), size)
        screen_pos = camera.apply_transform(None, (self.x, self.y))
        screen.blit(surface, (screen_pos[0] - size, screen_pos[1] - size))

class DamagePopup:
    def __init__(self, x, y, damage):
        self.x = x
        self.y = y
        self.damage = round(damage, 1)
        self.velocity_y = -20
        self.alpha = 255
        self.lifetime = 1.0
        self.start_time = tms.time()

    def update(self, delta_time):
        self.y += self.velocity_y * delta_time
        elapsed = tms.time() - self.start_time
        self.alpha = max(0, 255 * (1 - elapsed / self.lifetime))
        return elapsed < self.lifetime

    def draw(self, screen, camera):
        if self.alpha > 0:
            font = pygame.font.SysFont('arial', 20)
            text = font.render(str(self.damage), True, (255, 0, 0))
            text.set_alpha(int(self.alpha))
            screen_pos = camera.apply_transform(None, (self.x, self.y))
            screen.blit(text, (screen_pos[0] - text.get_width() // 2, screen_pos[1]))

class SmokeParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(5, 10)
        self.velocity_x = math.cos(angle) * speed
        self.velocity_y = math.sin(angle) * speed
        self.size = random.uniform(5, 10)
        self.alpha = 100
        self.lifetime = 1.0
        self.start_time = tms.time()

    def update(self, delta_time):
        self.x += self.velocity_x * delta_time
        self.y += self.velocity_y * delta_time
        elapsed = tms.time() - self.start_time
        self.alpha = max(0, 100 * (1 - elapsed / self.lifetime))
        return elapsed < self.lifetime

    def draw(self, screen, camera):
        if self.alpha > 0:
            surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (100, 100, 100, int(self.alpha)), (self.size, self.size), self.size)
            screen_pos = camera.apply_transform(None, (self.x, self.y))
            screen.blit(surface, (screen_pos[0] - self.size, screen_pos[1] - self.size))

class SparkParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(7.5, 15)
        self.velocity_x = math.cos(angle) * speed
        self.velocity_y = math.sin(angle) * speed
        self.length = random.uniform(4, 8)
        self.rotation = random.uniform(0, 2 * math.pi)
        self.angular_velocity = random.uniform(-2, 2)
        self.alpha = 255
        self.lifetime = 0.3
        self.start_time = tms.time()

    def update(self, delta_time):
        self.x += self.velocity_x * delta_time
        self.y += self.velocity_y * delta_time
        self.rotation += self.angular_velocity * delta_time
        elapsed = tms.time() - self.start_time
        self.alpha = max(0, 255 * (1 - elapsed / self.lifetime))
        return elapsed < self.lifetime

    def draw(self, screen, camera):
        if self.alpha > 0:
            surface = pygame.Surface((self.length * 2, self.length * 2), pygame.SRCALPHA)
            cos_rot = math.cos(self.rotation)
            sin_rot = math.sin(self.rotation)
            x1 = -self.length / 2 * cos_rot
            y1 = -self.length / 2 * sin_rot
            x2 = self.length / 2 * cos_rot
            y2 = self.length / 2 * sin_rot
            pygame.draw.line(surface, (255, 200, 0, int(self.alpha)), 
                           (self.length + x1, self.length + y1), 
                           (self.length + x2, self.length + y2), 1)
            screen_pos = camera.apply_transform(None, (self.x, self.y))
            screen.blit(surface, (screen_pos[0] - self.length, screen_pos[1] - self.length))

class NitroFlameParticle:
    def __init__(self, x, y, car_angle):
        self.x = x
        self.y = y
        angle = car_angle + math.pi + random.uniform(-0.2, 0.2)
        speed = random.uniform(10, 15)
        self.velocity_x = math.cos(angle) * speed
        self.velocity_y = math.sin(angle) * speed
        self.size = random.uniform(3, 6)
        self.alpha = 200
        self.lifetime = 0.4
        self.start_time = tms.time()

    def update(self, delta_time):
        self.x += self.velocity_x * delta_time
        self.y += self.velocity_y * delta_time
        elapsed = tms.time() - self.start_time
        self.alpha = max(0, 200 * (1 - elapsed / self.lifetime))
        return elapsed < self.lifetime

    def draw(self, screen, camera):
        if self.alpha > 0:
            surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (0, 191, 255, int(self.alpha)), (self.size, self.size), self.size)
            screen_pos = camera.apply_transform(None, (self.x, self.y))
            screen.blit(surface, (screen_pos[0] - self.size, screen_pos[1] - self.size))