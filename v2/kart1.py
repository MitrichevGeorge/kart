import pygame
import sys
import math
import requests
import json
import threading
import time
import uuid
import os
import random
from collections import deque

# Инициализация Pygame
pygame.init()

# Загрузка карты
map_image = pygame.image.load('map.png')
MAP_WIDTH, MAP_HEIGHT = map_image.get_size()
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Karting Game")

# Цвета карты
COLOR_FLOOR = (0, 0, 0)
COLOR_WALL = (200, 200, 200)
COLOR_SAND = (180, 180, 0)
COLOR_START = (50, 200, 0)

# Параметры машинки
CAR_WIDTH = 25
CAR_HEIGHT = 20
WHEEL_WIDTH = 8
WHEEL_HEIGHT = 6
FRONT_WHEEL_HEIGHT = 6
CAR_OTHER_COLOR = (255, 0, 0)
WHEEL_COLOR = (100, 100, 100)
WHEEL_ACTIVE_COLOR = (0, 255, 0)
TRAIL_COLOR = (80, 80, 80)
BURNT_COLOR = (50, 50, 50)

# Физические параметры
ACCELERATION = 0.3
DECELERATION = 0.04
MAX_SPEED = 10
TURN_ACCELERATION = 0.01
ROTATIONAL_FRICTION = 0.04
MAX_ANGULAR_VELOCITY = 0.40
SAND_SLOWDOWN = 0.9
SAND_INERTIA_LOSS = 0.08
WALL_BOUNCE = 0.3
FRICTION = 0.3
TRAIL_FADE_RATE = 0.99
MIN_SPEED_FOR_TURN = 0.5
LOW_SPEED_TURN_FACTOR = 0.3
HIGH_SPEED_DRIFT_FACTOR = 0.3
CAR_COLLISION_BOUNCE = 0.5
MIN_SPAWN_DISTANCE = 30
BLEND_FACTOR = 0.5
MAX_HEALTH = 20
DAMAGE_SCALING = 0.5
SPAWN_PROTECTION_TIME = 2.0
HEALTH_BAR_WIDTH = 40
HEALTH_BAR_HEIGHT = 6
HEALTH_BAR_OFFSET = 0  # Reduced from 40 to move closer
NAME_OFFSET = 30        # New constant for name position
SMOKE_HEALTH_THRESHOLD = 9
SMOKE_EMISSION_RATE = 0.1
SMOKE_LIFETIME = 1.0
SMOKE_SPEED = 10
POPUP_LIFETIME = 1.0
POPUP_SPEED = 20
EXPLOSION_LIFETIME = 0.5
EXPLOSION_SIZE = 40
CORPSE_LIFETIME = 3.0  # Time the burnt corpse persists

# Поверхность для следов
trail_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)

# Шрифты
font = pygame.font.SysFont('arial', 20)
font_large = pygame.font.SysFont('arial', 30)
font_small = pygame.font.SysFont('arial', 15)

# Network settings
SERVER_URL = 'http://geomit23.pythonanywhere.com/webhook'
PLAYER_ID = str(uuid.uuid4())
other_players = {}
network_lock = threading.Lock()
ping_times = deque(maxlen=5)
connection_attempts = 0
MAX_CONNECTION_ATTEMPTS = 3
connection_established = False
is_paused = False

# Predefined colors
COLOR_OPTIONS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 128)
]

class Explosion:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.start_time = time.time()
        self.lifetime = EXPLOSION_LIFETIME

    def update(self, delta_time):
        elapsed = time.time() - self.start_time
        return elapsed < self.lifetime

    def draw(self, screen, camera):
        if not self.update(0):
            return
        elapsed = time.time() - self.start_time
        alpha = max(0, 255 * (1 - elapsed / self.lifetime))
        size = EXPLOSION_SIZE * (elapsed / self.lifetime)
        surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (255, 100, 0, int(alpha)), (size, size), size)
        screen_pos = camera.apply_transform(None, (self.x, self.y))
        screen.blit(surface, (screen_pos[0] - size, screen_pos[1] - size))

class DamagePopup:
    def __init__(self, x, y, damage):
        self.x = x
        self.y = y
        self.damage = round(damage, 1)
        self.velocity_y = -POPUP_SPEED
        self.alpha = 255
        self.lifetime = POPUP_LIFETIME
        self.start_time = time.time()

    def update(self, delta_time):
        self.y += self.velocity_y * delta_time
        elapsed = time.time() - self.start_time
        self.alpha = max(0, 255 * (1 - elapsed / self.lifetime))
        return elapsed < self.lifetime

    def draw(self, screen, camera):
        if self.alpha > 0:
            text = font.render(str(self.damage), True, (255, 0, 0))
            text.set_alpha(int(self.alpha))
            screen_pos = camera.apply_transform(None, (self.x, self.y))
            screen.blit(text, (screen_pos[0] - text.get_width() // 2, screen_pos[1]))

class SmokeParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(SMOKE_SPEED * 0.5, SMOKE_SPEED)
        self.velocity_x = math.cos(angle) * speed
        self.velocity_y = math.sin(angle) * speed
        self.size = random.uniform(5, 10)
        self.alpha = 100
        self.lifetime = SMOKE_LIFETIME
        self.start_time = time.time()

    def update(self, delta_time):
        self.x += self.velocity_x * delta_time
        self.y += self.velocity_y * delta_time
        elapsed = time.time() - self.start_time
        self.alpha = max(0, 100 * (1 - elapsed / self.lifetime))
        return elapsed < self.lifetime

    def draw(self, screen, camera):
        if self.alpha > 0:
            surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (100, 100, 100, int(self.alpha)), (self.size, self.size), self.size)
            screen_pos = camera.apply_transform(None, (self.x, self.y))
            screen.blit(surface, (screen_pos[0] - self.size, screen_pos[1] - self.size))

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
        new_zoom = self.zoom + delta
        self.zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

    def apply_transform(self, surface, pos):
        screen_x = (pos[0] - self.x) * self.zoom + WINDOW_WIDTH / 2
        screen_y = (pos[1] - self.y) * self.zoom + WINDOW_HEIGHT / 2
        return screen_x, screen_y

    def apply_surface_transform(self, surface, pos):
        scaled_surface = pygame.transform.scale(
            surface,
            (int(surface.get_width() * self.zoom), int(surface.get_height() * self.zoom))
        )
        screen_x = (pos[0] - self.x) * self.zoom + WINDOW_WIDTH / 2
        screen_y = (pos[1] - self.y) * self.zoom + WINDOW_HEIGHT / 2
        return scaled_surface, (screen_x, screen_y)

def is_valid_color(color):
    r, g, b = color
    return r + g + b >= 150 and max(r, g, b) >= 50

def load_config():
    config_file = '.kart_config.json'
    default_config = {'name': 'Player', 'color': [128, 128, 128]}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if (isinstance(config.get('name'), str) and
                    isinstance(config.get('color'), list) and
                    len(config['color']) == 3 and
                    is_valid_color(config['color'])):
                    return config
        except (json.JSONDecodeError, KeyError):
            pass
    return default_config

def save_config(name, color):
    config_file = '.kart_config.json'
    config = {'name': name, 'color': color}
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f)
    except IOError as e:
        print(f"Failed to save config: {e}")

def draw_health_bar(screen, camera, x, y, health, max_health):
    health_ratio = max(0, min(1, health / max_health))
    bar_width = HEALTH_BAR_WIDTH * health_ratio
    green = int(255 * health_ratio)
    red = int(255 * (1 - health_ratio))
    color = (red, green, 0)
    
    center_x, center_y = x - CAR_WIDTH // 2 + HEALTH_BAR_WIDTH // 2, y - CAR_HEIGHT - HEALTH_BAR_OFFSET
    screen_pos = camera.apply_transform(None, (center_x, center_y))
    
    bg_rect = pygame.Rect(screen_pos[0] - HEALTH_BAR_WIDTH // 2, screen_pos[1] - HEALTH_BAR_HEIGHT // 2, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT)
    pygame.draw.rect(screen, (50, 50, 50), bg_rect, border_radius=HEALTH_BAR_HEIGHT // 2)
    
    if health_ratio > 0:
        health_rect = pygame.Rect(screen_pos[0] - HEALTH_BAR_WIDTH // 2, screen_pos[1] - HEALTH_BAR_HEIGHT // 2, bar_width, HEALTH_BAR_HEIGHT)
        pygame.draw.rect(screen, color, health_rect, border_radius=HEALTH_BAR_HEIGHT // 2)

def render_text_with_outline(text, font, color, pos, camera=None):
    inv_color = (255 - color[0], 255 - color[1], 255 - color[2])
    text_surface = font.render(text, True, color)
    outline_surface = font.render(text, True, inv_color)
    if camera:
        screen_pos = camera.apply_transform(None, pos)
    else:
        screen_pos = pos
    outline_positions = [
        (screen_pos[0] - 1, screen_pos[1]),
        (screen_pos[0] + 1, screen_pos[1]),
        (screen_pos[0], screen_pos[1] - 1),
        (screen_pos[0], screen_pos[1] + 1)
    ]
    for outline_pos in outline_positions:
        screen.blit(outline_surface, outline_pos)
    screen.blit(text_surface, screen_pos)

def show_start_screen():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    config = load_config()
    input_name = config['name']
    selected_color = config['color']
    input_active = False
    cursor = '_'
    cursor_timer = 0
    cursor_visible = True

    button_width = 100
    button_height = 40

    while True:
        play_button = pygame.Rect(WINDOW_WIDTH // 2 - button_width // 2, WINDOW_HEIGHT - 100, button_width, button_height)
        color_buttons = [
            pygame.Rect(WINDOW_WIDTH // 2 - len(COLOR_OPTIONS) * 40 // 2 + i * 40, WINDOW_HEIGHT // 2 + 50, 30, 30)
            for i in range(len(COLOR_OPTIONS))
        ]
        name_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 50, 200, 30)

        screen.fill((50, 50, 50))
        
        title = font_large.render("Karting Game Setup", True, (255, 255, 255))
        screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 50))

        name_label = font.render("Name:", True, (255, 255, 255))
        screen.blit(name_label, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 50))
        name_surface = font.render(input_name + (cursor if input_active and cursor_visible else ''), True, (255, 255, 255))
        pygame.draw.rect(screen, (255, 255, 255), name_rect, 2)
        screen.blit(name_surface, (name_rect.x + 5, name_rect.y + 5))

        color_label = font.render("Car Color:", True, (255, 255, 255))
        screen.blit(name_label, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2))
        for i, button in enumerate(color_buttons):
            pygame.draw.rect(screen, COLOR_OPTIONS[i], button)
            if list(selected_color) == list(COLOR_OPTIONS[i]):
                pygame.draw.rect(screen, (255, 255, 255), button, 2)

        pygame.draw.rect(screen, (0, 200, 0), play_button)
        play_text = font.render("Play", True, (255, 255, 255))
        screen.blit(play_text, (play_button.x + (button_width - play_text.get_width()) // 2,
                               play_button.y + (button_height - play_text.get_height()) // 2))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if play_button.collidepoint(event.pos):
                    save_config(input_name, selected_color)
                    return input_name, selected_color
                if name_rect.collidepoint(event.pos):
                    input_active = True
                else:
                    input_active = False
                for i, button in enumerate(color_buttons):
                    if button.collidepoint(event.pos):
                        selected_color = COLOR_OPTIONS[i]
            elif event.type == pygame.KEYDOWN and input_active:
                if event.key == pygame.K_BACKSPACE:
                    input_name = input_name[:-1]
                elif event.key == pygame.K_RETURN:
                    input_active = False
                elif event.unicode.isalnum() or event.unicode == ' ':
                    if len(input_name) < 20:
                        input_name += event.unicode

        cursor_timer += 1
        if cursor_timer >= 30:
            cursor_visible = not cursor_visible
            cursor_timer = 0

def show_connection_screen():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT, connection_established
    dots = ""
    start_time = time.time()
    while not connection_established:
        screen.fill((50, 50, 50))
        
        elapsed = time.time() - start_time
        dots = "." * (int(elapsed * 2) % 4)
        
        text = font_large.render(f"Connecting to server{dots}", True, (255, 255, 255))
        screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        
        time.sleep(0.1)

def show_connection_lost_screen():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    text = font_large.render("Connection to server...", True, (255, 255, 255))
    screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2))

def show_death_screen():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((100, 0, 0, 128))
    screen.blit(overlay, (0, 0))
    
    death_text = font_large.render("You Died", True, (255, 255, 255))
    respawn_text = font_small.render("Click to respawn", True, (255, 255, 255))
    
    screen.blit(death_text, (WINDOW_WIDTH // 2 - death_text.get_width() // 2, WINDOW_HEIGHT // 2 - 50))
    screen.blit(respawn_text, (WINDOW_WIDTH // 2 - respawn_text.get_width() // 2, WINDOW_HEIGHT // 2 + 20))

def network_thread(local_car, player_name, player_color):
    global other_players, ping_times, connection_attempts, connection_established, is_paused
    while True:
        if is_paused and connection_established:
            time.sleep(0.1)
            continue
        try:
            start_time = time.time()
            state = {
                'x': local_car.x,
                'y': local_car.y,
                'angle': local_car.angle,
                'speed': local_car.speed,
                'steering_angle': local_car.steering_angle,
                'velocity_x': local_car.velocity_x,
                'velocity_y': local_car.velocity_y,
                'angular_velocity': local_car.angular_velocity,
                'checkpoints_passed': local_car.checkpoints_passed,
                'health': local_car.health,
                'is_dead': local_car.is_dead,
                'death_time': local_car.death_time if local_car.is_dead else 0
            }
            payload = {
                'player_id': PLAYER_ID,
                'state': state,
                'name': player_name,
                'color': player_color
            }
            response = requests.post(SERVER_URL, json=payload, timeout=1)
            end_time = time.time()
            
            ping_ms = (end_time - start_time) * 1000
            ping_times.append(ping_ms)
            
            if response.status_code == 200:
                connection_attempts = 0
                connection_established = True
                is_paused = False
                with network_lock:
                    new_states = response.json()
                    current_time = time.time()
                    for pid, data in new_states.items():
                        if pid != PLAYER_ID:
                            state = data['state']
                            name = data['name']
                            color = data['color']
                            if pid in other_players:
                                car = other_players[pid]['car']
                                car.x = BLEND_FACTOR * car.x + (1 - BLEND_FACTOR) * state['x']
                                car.y = BLEND_FACTOR * car.y + (1 - BLEND_FACTOR) * state['y']
                                car.angle = BLEND_FACTOR * car.angle + (1 - BLEND_FACTOR) * state['angle']
                                car.speed = BLEND_FACTOR * car.speed + (1 - BLEND_FACTOR) * state['speed']
                                car.steering_angle = BLEND_FACTOR * car.steering_angle + (1 - BLEND_FACTOR) * state['steering_angle']
                                car.velocity_x = BLEND_FACTOR * car.velocity_x + (1 - BLEND_FACTOR) * state['velocity_x']
                                car.velocity_y = BLEND_FACTOR * car.velocity_y + (1 - BLEND_FACTOR) * state['velocity_y']
                                car.angular_velocity = BLEND_FACTOR * car.angular_velocity + (1 - BLEND_FACTOR) * state['angular_velocity']
                                car.health = state.get('health', MAX_HEALTH)
                                car.is_dead = state.get('is_dead', False)
                                car.death_time = state.get('death_time', 0)
                            else:
                                car = Car(state['x'], state['y'], state['angle'], is_local_player=False)
                                car.velocity_x = state['velocity_x']
                                car.velocity_y = state['velocity_y']
                                car.speed = state['speed']
                                car.steering_angle = state['steering_angle']
                                car.angular_velocity = state['angular_velocity']
                                car.health = state.get('health', MAX_HEALTH)
                                car.is_dead = state.get('is_dead', False)
                                car.death_time = state.get('death_time', 0)
                            car.checkpoints_passed = state['checkpoints_passed']
                            car.color = color
                            car.name = name
                            other_players[pid] = {'car': car, 'last_update': current_time}
                    other_players_copy = other_players.copy()
                    for pid in other_players_copy:
                        if pid not in new_states and pid != PLAYER_ID:
                            del other_players[pid]
            else:
                connection_attempts += 1
                if connection_attempts >= MAX_CONNECTION_ATTEMPTS:
                    is_paused = True
        except requests.RequestException:
            connection_attempts += 1
            if connection_attempts >= MAX_CONNECTION_ATTEMPTS:
                is_paused = True
        time.sleep(1/30)

class Car:
    def __init__(self, x, y, angle, render_enabled=True, training_mode=False, is_local_player=True):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.velocity_x = 0
        self.velocity_y = 0
        self.steering_angle = 0
        self.last_speed = 0
        self.angular_velocity = 0
        self.is_accelerating = False
        self.is_braking = False
        self.is_turning_left = False
        self.is_turning_right = False
        self.checkpoints_passed = 0
        self.render_enabled = render_enabled
        self.training_mode = training_mode
        self.is_local_player = is_local_player
        self.color = [128, 128, 128] if is_local_player else CAR_OTHER_COLOR
        self.name = "Player" if is_local_player else ""
        self.wheel_positions = [
            (-CAR_WIDTH // 2 + 5, CAR_HEIGHT // 2),
            (-CAR_WIDTH // 2 + 5, -CAR_HEIGHT // 2),
            (CAR_WIDTH // 2 - 5, CAR_HEIGHT // 2),
            (CAR_WIDTH // 2 - 5, -CAR_HEIGHT // 2)
        ]
        self.health = MAX_HEALTH
        self.spawn_protection = True
        self.spawn_time = time.time()
        self.spawn_x = x
        self.spawn_y = y
        self.damage_popups = []
        self.smoke_particles = []
        self.last_smoke_time = 0
        self.is_dead = False
        self.death_time = 0
        self.explosion = None

    def update(self, keys=None):
        if self.is_dead and (time.time() - self.death_time > CORPSE_LIFETIME):
            if self.is_local_player:
                self.is_dead = False  # Allow local player to respawn via click
            else:
                return  # Non-local players will be removed by network cleanup
        if self.is_dead:
            if self.explosion:
                self.explosion.update(1/60)
            return

        delta_time = 1/60
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        surface_color = get_surface_color(self.x, self.y)

        accel = 0
        turn_input = 0
        speed_factor = abs(self.speed) / MAX_SPEED

        if self.is_local_player and keys:
            self.is_accelerating = keys[pygame.K_UP]
            self.is_braking = keys[pygame.K_DOWN]
            self.is_turning_left = keys[pygame.K_LEFT]
            self.is_turning_right = keys[pygame.K_RIGHT]
        
        if self.is_accelerating:
            accel = ACCELERATION
        if self.is_braking:
            accel = -ACCELERATION * 0.5
        if self.is_turning_left:
            turn_input = -TURN_ACCELERATION
            self.steering_angle = max(self.steering_angle - TURN_ACCELERATION, -math.pi / 6)
        elif self.is_turning_right:
            turn_input = TURN_ACCELERATION
            self.steering_angle = min(self.steering_angle + TURN_ACCELERATION, math.pi / 6)
        else:
            self.steering_angle *= 0.8

        if surface_color == COLOR_SAND:
            accel *= SAND_SLOWDOWN
            turn_input *= SAND_SLOWDOWN
            self.speed *= (1 - SAND_INERTIA_LOSS)
            self.velocity_x *= (1 - SAND_INERTIA_LOSS)
            self.velocity_y *= (1 - SAND_INERTIA_LOSS)
            self.angular_velocity *= (1 - SAND_INERTIA_LOSS)

        self.last_speed = self.speed
        self.speed += accel
        self.speed = max(min(self.speed, MAX_SPEED), -MAX_SPEED / 2)
        if abs(self.speed) < DECELERATION and not self.is_accelerating and not self.is_braking:
            self.speed = 0
        else:
            self.speed *= (1 - DECELERATION)

        if abs(self.speed) > MIN_SPEED_FOR_TURN:
            turn_scale = LOW_SPEED_TURN_FACTOR + (1 - LOW_SPEED_TURN_FACTOR) * speed_factor
            self.angular_velocity += turn_input * turn_scale
            max_angular = MAX_ANGULAR_VELOCITY * turn_scale
            self.angular_velocity = max(min(self.angular_velocity, max_angular), -max_angular)
        else:
            self.angular_velocity *= 0.5

        if not self.is_turning_left and not self.is_turning_right:
            self.angular_velocity *= (1 - ROTATIONAL_FRICTION)

        self.angle += self.angular_velocity * (1 - speed_factor * 0.5)

        drift_scale = 1 - speed_factor * HIGH_SPEED_DRIFT_FACTOR
        if speed_factor > 0.8 and abs(self.angular_velocity) > 0.01:
            self.angle += self.angular_velocity * speed_factor * 0.2

        direction_x = cos_angle
        direction_y = sin_angle
        current_drift_factor = 1 - speed_factor * HIGH_SPEED_DRIFT_FACTOR
        self.velocity_x = self.velocity_x * current_drift_factor + direction_x * self.speed * (1 - current_drift_factor)
        self.velocity_y = self.velocity_y * current_drift_factor + direction_y * self.speed * (1 - current_drift_factor)

        new_x = self.x + self.velocity_x
        new_y = self.y + self.velocity_y
        if get_surface_color(new_x, new_y) == COLOR_WALL:
            old_velocity = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
            self.velocity_x *= -WALL_BOUNCE
            self.velocity_y *= -WALL_BOUNCE
            new_velocity = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
            impulse = old_velocity - new_velocity
            damage = impulse * DAMAGE_SCALING
            if damage > 0.1:
                self.damage_popups.append(DamagePopup(self.x, self.y, damage))
            self.health = max(0, self.health - damage)
            self.speed *= WALL_BOUNCE
            self.angular_velocity *= WALL_BOUNCE
        else:
            self.x = new_x
            self.y = new_y

        if self.health <= 0 and not self.is_dead:
            self.is_dead = True
            self.death_time = time.time()
            self.explosion = Explosion(self.x, self.y)
            self.velocity_x = 0
            self.velocity_y = 0
            self.speed = 0
            self.angular_velocity = 0

        distance_from_spawn = math.sqrt((self.x - self.spawn_x)**2 + (self.y - self.spawn_y)**2)
        if (distance_from_spawn > MIN_SPAWN_DISTANCE or
            time.time() - self.spawn_time > SPAWN_PROTECTION_TIME):
            self.spawn_protection = False

        if self.health <= SMOKE_HEALTH_THRESHOLD and not self.training_mode and not self.is_dead:
            current_time = time.time()
            if current_time - self.last_smoke_time >= SMOKE_EMISSION_RATE:
                self.smoke_particles.append(SmokeParticle(self.x, self.y))
                self.last_smoke_time = current_time

        self.damage_popups = [popup for popup in self.damage_popups if popup.update(delta_time)]
        self.smoke_particles = [particle for particle in self.smoke_particles if particle.update(delta_time)]

        if self.render_enabled and not self.training_mode and not self.is_dead:
            self.draw_trails()

    def draw_trails(self):
        relative_speed = abs(self.speed) + abs(self.steering_angle * self.speed)
        trail_alpha = min(int(relative_speed / MAX_SPEED * 255 * FRICTION * 2), 255)
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        for i, (wx, wy) in enumerate(self.wheel_positions):
            wheel_x = self.x + wx * cos_angle - wy * sin_angle
            wheel_y = self.y + wx * sin_angle + wy * cos_angle
            if i >= 2 and abs(self.steering_angle) > 0.01:
                trail_alpha = min(trail_alpha * 1.5, 255)
            if trail_alpha > 5:
                pygame.draw.circle(trail_surface, (*TRAIL_COLOR, trail_alpha), (int(wheel_x), int(wheel_y)), 3)

    def draw(self, camera):
        if not self.render_enabled or self.training_mode:
            return
        if self.is_dead and (time.time() - self.death_time > CORPSE_LIFETIME):
            return
        points = [
            (-CAR_WIDTH // 2, -CAR_HEIGHT // 2),
            (CAR_WIDTH // 2, -CAR_HEIGHT // 2),
            (CAR_WIDTH // 2, CAR_HEIGHT // 2),
            (-CAR_WIDTH // 2, CAR_HEIGHT // 2)
        ]
        rotated_points = []
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        color = BURNT_COLOR if self.is_dead else self.color
        for x, y in points:
            rx = x * cos_angle - y * sin_angle
            ry = x * sin_angle + y * cos_angle
            screen_pos = camera.apply_transform(None, (self.x + rx, self.y + ry))
            rotated_points.append(screen_pos)
        pygame.draw.polygon(screen, color, rotated_points)

        if not self.is_dead:
            for i, (wx, wy) in enumerate(self.wheel_positions):
                wheel_angle = self.angle
                wheel_h = WHEEL_HEIGHT if i < 2 else FRONT_WHEEL_HEIGHT
                if i >= 2:
                    wheel_angle += self.steering_angle
                wheel_points = [
                    (-WHEEL_WIDTH // 2, -wheel_h // 2),
                    (WHEEL_WIDTH // 2, -wheel_h // 2),
                    (WHEEL_WIDTH // 2, wheel_h // 2),
                    (-WHEEL_WIDTH // 2, wheel_h // 2)
                ]
                rotated_wheel = []
                cos_wheel = math.cos(wheel_angle)
                sin_wheel = math.sin(wheel_angle)
                wheel_x = self.x + wx * cos_angle - wy * sin_angle
                wheel_y = self.y + wx * sin_angle + wy * cos_angle
                for x, y in wheel_points:
                    rx = x * cos_wheel - y * sin_wheel
                    ry = x * sin_wheel + y * cos_wheel
                    screen_pos = camera.apply_transform(None, (wheel_x + rx, wheel_y + ry))
                    rotated_wheel.append(screen_pos)
                color = WHEEL_ACTIVE_COLOR if (i < 2 and (self.is_accelerating or self.is_braking)) or (i >= 2 and (self.is_turning_left or self.is_turning_right)) else WHEEL_COLOR
                pygame.draw.polygon(screen, color, rotated_wheel)

        if self.explosion:
            self.explosion.draw(screen, camera)

        if not self.is_dead:
            render_text_with_outline(self.name, font, (255, 255, 255), (self.x - CAR_WIDTH // 2, self.y - CAR_HEIGHT - NAME_OFFSET), camera)
            draw_health_bar(screen, camera, self.x, self.y, self.health, MAX_HEALTH)

        for popup in self.damage_popups:
            popup.draw(screen, camera)
        for particle in self.smoke_particles:
            particle.draw(screen, camera)

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.velocity_x = 0
        self.velocity_y = 0
        self.steering_angle = 0
        self.last_speed = 0
        self.angular_velocity = 0
        self.checkpoints_passed = 0
        self.total_reward = 0
        self.health = MAX_HEALTH
        self.spawn_protection = True
        self.spawn_time = time.time()
        self.spawn_x = x
        self.spawn_y = y
        self.damage_popups = []
        self.smoke_particles = []
        self.last_smoke_time = 0
        self.is_dead = False
        self.death_time = 0
        self.explosion = None

def get_surface_color(x, y):
    if 0 <= x < MAP_WIDTH and 0 <= y < MAP_HEIGHT:
        return map_image.get_at((int(x), int(y)))[:3]
    return COLOR_WALL

def find_start_position():
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            if map_image.get_at((x, y))[:3] == COLOR_START:
                return x, y
    return MAP_WIDTH // 2, MAP_HEIGHT // 2

def check_collision(car1, car2):
    if car1.spawn_protection or car2.spawn_protection or car1.is_dead or car2.is_dead:
        return

    rect1 = pygame.Rect(car1.x - CAR_WIDTH // 2, car1.y - CAR_HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT)
    rect2 = pygame.Rect(car2.x - CAR_WIDTH // 2, car2.y - CAR_HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT)
    
    if rect1.colliderect(rect2):
        dx = car1.x - car2.x
        dy = car1.y - car2.y
        distance = max(math.sqrt(dx**2 + dy**2), 0.1)
        
        nx = dx / distance
        ny = dy / distance
        
        rvx = car1.velocity_x - car2.velocity_x
        rvy = car1.velocity_y - car2.velocity_y
        
        dot = rvx * nx + rvy * ny
        
        if dot > 0:
            impulse = (2 * dot) / 2
            impulse_magnitude = abs(impulse)
            
            car1.velocity_x -= impulse * nx * CAR_COLLISION_BOUNCE
            car1.velocity_y -= impulse * ny * CAR_COLLISION_BOUNCE
            car2.velocity_x += impulse * nx * CAR_COLLISION_BOUNCE
            car2.velocity_y += impulse * ny * CAR_COLLISION_BOUNCE
            
            damage = impulse_magnitude * DAMAGE_SCALING
            if damage > 0.1:
                car1.damage_popups.append(DamagePopup(car1.x, car1.y, damage))
                car2.damage_popups.append(DamagePopup(car2.x, car2.y, damage))
            car1.health = max(0, car1.health - damage)
            car2.health = max(0, car2.health - damage)
            
            car1.speed *= CAR_COLLISION_BOUNCE
            car2.speed *= CAR_COLLISION_BOUNCE
            car1.angular_velocity *= CAR_COLLISION_BOUNCE
            car2.angular_velocity *= CAR_COLLISION_BOUNCE
            
            overlap = (CAR_WIDTH + CAR_HEIGHT) / 2 - distance
            if overlap > 0:
                car1.x += nx * overlap / 2
                car1.y += ny * overlap / 2
                car2.x -= nx * overlap / 2
                car2.y -= ny * overlap / 2

# Start screen
player_name, player_color = show_start_screen()

# Initialize local player
start_x, start_y = find_start_position()
local_car = Car(start_x, start_y, 0, is_local_player=True)
local_car.name = player_name
local_car.color = player_color

# Initialize camera
camera = Camera()
camera.x = start_x
camera.y = start_y

# Start network thread
network_thread = threading.Thread(target=network_thread, args=(local_car, player_name, player_color), daemon=True)
network_thread.start()

# Wait for server connection
show_connection_screen()

def main():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT, is_paused
    clock = pygame.time.Clock()
    FPS = 60
    last_time = time.time()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEWHEEL:
                camera.adjust_zoom(event.y * 0.1)
            elif event.type == pygame.VIDEORESIZE:
                WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN and local_car.is_dead:
                start_x, start_y = find_start_position()
                local_car.reset(start_x, start_y)
                camera.x = start_x
                camera.y = start_y

        if is_paused:
            screen.fill((0, 0, 0))
            scaled_map, map_pos = camera.apply_surface_transform(map_image, (0, 0))
            screen.blit(scaled_map, map_pos)
            scaled_trails, trails_pos = camera.apply_surface_transform(trail_surface, (0, 0))
            screen.blit(scaled_trails, trails_pos)
            
            with network_lock:
                for pid, data in other_players.items():
                    if pid != PLAYER_ID:
                        data['car'].draw(camera)
            
            local_car.draw(camera)
            show_connection_lost_screen()
            pygame.display.flip()
            clock.tick(FPS)
            continue

        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        keys = pygame.key.get_pressed()
        local_car.update(keys)

        camera.update(local_car.x, local_car.y)

        with network_lock:
            for pid, data in other_players.items():
                if pid != PLAYER_ID:
                    car = data['car']
                    car.update()
                    data['last_update'] = current_time

        with network_lock:
            for pid, data in other_players.items():
                if pid != PLAYER_ID:
                    check_collision(local_car, data['car'])

        faded_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)
        faded_surface.blit(trail_surface, (0, 0))
        faded_surface.set_alpha(int(255 * TRAIL_FADE_RATE))
        trail_surface.fill((0, 0, 0, 0))
        trail_surface.blit(faded_surface, (0, 0))

        screen.fill((0, 0, 0))

        scaled_map, map_pos = camera.apply_surface_transform(map_image, (0, 0))
        screen.blit(scaled_map, map_pos)
        scaled_trails, trails_pos = camera.apply_surface_transform(trail_surface, (0, 0))
        screen.blit(scaled_trails, trails_pos)
        
        with network_lock:
            for pid, data in other_players.items():
                if pid != PLAYER_ID:
                    data['car'].draw(camera)
        
        local_car.draw(camera)

        if local_car.is_dead:
            show_death_screen()

        avg_ping = sum(ping_times) / len(ping_times) if ping_times else 0
        ping_text = f"Ping: {int(avg_ping)} ms"
        ping_surface = font.render(ping_text, True, (255, 255, 255))
        ping_pos = (WINDOW_WIDTH - ping_surface.get_width() - 10, 10)
        render_text_with_outline(ping_text, font, (255, 255, 255), ping_pos, camera=None)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()