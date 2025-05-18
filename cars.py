import pygame
import numpy as np
import torch
import torch.nn as nn
import json
import math
import asyncio
import platform

# Инициализация Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
FPS = 60

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Класс машинки
class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.alive = True
        self.time = 0
        self.finished = False
        self.size = 20

    def move(self, accel, turn):
        self.speed += accel * 0.1
        self.speed = np.clip(self.speed, -2, 4)
        self.angle += turn * 0.05
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
        self.time += 1

    def get_rays(self, walls):
        rays = []
        angles = [self.angle + a for a in [-math.pi/4, 0, math.pi/4, -math.pi/2, math.pi/2]]
        for a in angles:
            dist = self.cast_ray(a, walls)
            rays.append(dist)
        return np.array(rays) / 200.0  # Нормализация

    def cast_ray(self, angle, walls):
        x1, y1 = self.x, self.y
        x2 = x1 + 200 * math.cos(angle)
        y2 = y1 + 200 * math.sin(angle)
        min_dist = 200
        for wall in walls:
            x3, y3, x4, y4 = wall
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            if 0 < t < 1 and 0 < u < 1:
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                dist = math.sqrt((px - x1)**2 + (py - y1)**2)
                min_dist = min(min_dist, dist)
        return min_dist

    def check_collision(self, walls):
        for wall in walls:
            x3, y3, x4, y4 = wall
            for dx, dy in [(-self.size/2, -self.size/2), (self.size/2, -self.size/2),
                          (self.size/2, self.size/2), (-self.size/2, self.size/2)]:
                x, y = self.x + dx, self.y + dy
                denom = (self.x - x) * (y3 - y4) - (self.y - y) * (x3 - x4)
                if denom == 0:
                    continue
                t = ((self.x - x3) * (y3 - y4) - (self.y - y3) * (x3 - x4)) / denom
                u = -((self.x - x) * (self.y - y3) - (self.y - y) * (self.x - x3)) / denom
                if 0 < t < 1 and 0 < u < 1:
                    self.alive = False
                    return
        if self.finished:
            self.alive = False

    def check_finish(self, finish):
        fx, fy, fr = finish
        if math.sqrt((self.x - fx)**2 + (self.y - fy)**2) < fr:
            self.finished = True

    def draw(self, surface, camera, alpha=255):
        cx, cy, scale = camera
        px = (self.x - cx) * scale + WIDTH / 2
        py = (self.y - cy) * scale + HEIGHT / 2
        points = [
            (px + self.size * scale * math.cos(self.angle), py + self.size * scale * math.sin(self.angle)),
            (px + self.size * scale * math.cos(self.angle + 2.5), py + self.size * scale * math.sin(self.angle + 2.5)),
            (px + self.size * scale * math.cos(self.angle + math.pi), py + self.size * scale * math.sin(self.angle + math.pi)),
            (px + self.size * scale * math.cos(self.angle - 2.5), py + self.size * scale * math.sin(self.angle - 2.5)),
        ]
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(surf, (255, 0, 0, alpha), points)
        surface.blit(surf, (0, 0))

# Нейронная сеть
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# Эволюционный алгоритм
class Evolution:
    def __init__(self, population_size=20):
        self.population_size = population_size
        self.population = [NeuralNetwork() for _ in range(population_size)]
        self.scores = [0] * population_size
        self.generation = 0

    def evaluate(self, start, walls, finish):
        cars = [Car(start[0], start[1]) for _ in range(self.population_size)]
        max_steps = 1000
        for step in range(max_steps):
            for i, car in enumerate(cars):
                if not car.alive:
                    continue
                rays = car.get_rays(walls)
                inputs = torch.tensor(rays, dtype=torch.float32)
                with torch.no_grad():
                    outputs = self.population[i](inputs)
                accel, turn = outputs.numpy()
                car.move(accel, turn)
                car.check_collision(walls)
                car.check_finish(finish)
                if car.finished:
                    self.scores[i] = 10000 - car.time
                elif not car.alive:
                    self.scores[i] = -math.sqrt((car.x - finish[0])**2 + (car.y - finish[1])**2)
            if all(not car.alive for car in cars):
                break
        return cars

    def evolve(self):
        self.generation += 1
        # Сортировка по скор
        sorted_indices = np.argsort(self.scores)[::-1]
        self.scores = [self.scores[i] for i in sorted_indices]
        self.population = [self.population[i] for i in sorted_indices]
        # Элитизм: сохраняем лучших
        new_population = self.population[:2]
        # Кроссовер и мутации
        while len(new_population) < self.population_size:
            parent1 = self.population[np.random.randint(0, 5)]
            parent2 = self.population[np.random.randint(0, 5)]
            child = NeuralNetwork()
            for p1, p2, c in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
                mask = torch.rand_like(p1) > 0.5
                c.data = torch.where(mask, p1.data, p2.data)
                c.data += torch.randn_like(c.data) * 0.1
            new_population.append(child)
        self.population = new_population
        self.scores = [0] * self.population_size

# Основной игровой класс
class Game:
    def __init__(self):
        self.walls = []
        self.start = (100, 100)
        self.finish = (700, 500, 20)
        self.mode = 'edit'  # 'edit' или 'train'
        self.drawing = False
        self.erasing = False
        self.draw_start = None
        self.camera = [0, 0, 1.0]  # x, y, scale
        self.dragging = False
        self.evolution = Evolution()
        self.cars = []
        self.map_data = ""

    def save_map(self):
        map_data = {
            'walls': self.walls,
            'start': self.start,
            'finish': self.finish
        }
        self.map_data = json.dumps(map_data)
        return self.map_data

    def load_map(self, map_data):
        if map_data:
            data = json.loads(map_data)
            self.walls = data['walls']
            self.start = data['start']
            self.finish = data['finish']

    def screen_to_world(self, pos):
        cx, cy, scale = self.camera
        x = (pos[0] - WIDTH / 2) / scale + cx
        y = (pos[1] - HEIGHT / 2) / scale + cy
        return x, y

    def world_to_screen(self, pos):
        cx, cy, scale = self.camera
        x = (pos[0] - cx) * scale + WIDTH / 2
        y = (pos[1] - cy) * scale + HEIGHT / 2
        return x, y

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = self.screen_to_world(event.pos)
            if event.button == 1 and self.mode == 'edit':  # ЛКМ
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:  # Установка старта
                    self.start = pos
                elif pygame.key.get_mods() & pygame.KMOD_CTRL:  # Установка финиша
                    self.finish = (pos[0], pos[1], 20)
                else:
                    self.drawing = True
                    self.draw_start = pos
            elif event.button == 3:  # ПКМ
                self.erasing = True
            elif event.button == 2:  # СКМ
                self.dragging = True
                self.drag_start = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.drawing:
                self.drawing = False
                end_pos = self.screen_to_world(event.pos)
                self.walls.append([self.draw_start[0], self.draw_start[1], end_pos[0], end_pos[1]])
            elif event.button == 3:
                self.erasing = False
            elif event.button == 2:
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.mode == 'edit':
            if self.erasing:
                pos = self.screen_to_world(event.pos)
                self.walls = [w for w in self.walls if not self.point_near_wall(pos, w)]
            elif self.dragging:
                dx = event.pos[0] - self.drag_start[0]
                dy = event.pos[1] - self.drag_start[1]
                self.camera[0] -= dx / self.camera[2]
                self.camera[1] -= dy / self.camera[2]
                self.drag_start = event.pos
        elif event.type == pygame.MOUSEWHEEL:
            self.camera[2] *= 1.1 if event.y > 0 else 0.9
            self.camera[2] = max(0.1, min(self.camera[2], 10))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                self.save_map()
            elif event.key == pygame.K_t:
                self.mode = 'train'
                self.cars = self.evolution.evaluate(self.start, self.walls, self.finish)
            elif event.key == pygame.K_e:
                self.mode = 'edit'

    def point_near_wall(self, point, wall, threshold=10):
        x, y = point
        x1, y1, x2, y2 = wall
        denom = ((x2 - x1)**2 + (y2 - y1)**2)
        if denom == 0:
            return math.sqrt((x - x1)**2 + (y - y1)**2) < threshold
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / denom))
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return math.sqrt((x - px)**2 + (y - py)**2) < threshold

    def draw(self):
        screen.fill(WHITE)
        # Отрисовка стен
        for wall in self.walls:
            x1, y1 = self.world_to_screen((wall[0], wall[1]))
            x2, y2 = self.world_to_screen((wall[2], wall[3]))
            pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), 2)
        # Отрисовка старта
        sx, sy = self.world_to_screen(self.start)
        pygame.draw.circle(screen, GREEN, (sx, sy), 10)
        # Отрисовка финиша
        fx, fy = self.world_to_screen((self.finish[0], self.finish[1]))
        pygame.draw.circle(screen, RED, (fx, fy), self.finish[2] * self.camera[2])
        # Отрисовка машин
        if self.mode == 'train':
            for car in self.cars:
                car.draw(screen, self.camera, alpha=50)

async def main():
    game = Game()
    while True:
        for event in pygame.event.get():
            game.handle_event(event)
        game.draw()
        pygame.display.flip()
        clock.tick(FPS)
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())