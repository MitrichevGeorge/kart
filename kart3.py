import pygame
import sys
import math
import requests
import json
import threading
import time
import uuid
import os
from collections import deque

# Инициализация Pygame
pygame.init()

# Загрузка карты
map_image = pygame.image.load('map.png')
MAP_WIDTH, MAP_HEIGHT = map_image.get_size()
screen = pygame.display.set_mode((MAP_WIDTH, MAP_HEIGHT))
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

# Поверхность для следов
trail_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)

# Шрифт
font = pygame.font.SysFont('arial', 20)
font_large = pygame.font.SysFont('arial', 30)

# Network settings
SERVER_URL = 'http://geomit22.pythonanywhere.com/webhook'
PLAYER_ID = str(uuid.uuid4())
other_players = {}
network_lock = threading.Lock()
ping_times = deque(maxlen=5)

# Predefined colors (avoiding near-black)
COLOR_OPTIONS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 128, 128) # Gray
]

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

def render_text_with_outline(text, font, color, pos):
    inv_color = (255 - color[0], 255 - color[1], 255 - color[2])
    text_surface = font.render(text, True, color)
    outline_surface = font.render(text, True, inv_color)
    outline_positions = [
        (pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]),
        (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)
    ]
    for outline_pos in outline_positions:
        screen.blit(outline_surface, outline_pos)
    screen.blit(text_surface, pos)

def show_start_screen():
    config = load_config()
    input_name = config['name']
    selected_color = config['color']
    input_active = False
    cursor = '_'
    cursor_timer = 0
    cursor_visible = True

    button_width = 100
    button_height = 40
    play_button = pygame.Rect(MAP_WIDTH // 2 - button_width // 2, MAP_HEIGHT - 100, button_width, button_height)
    color_buttons = [
        pygame.Rect(MAP_WIDTH // 2 - len(COLOR_OPTIONS) * 40 // 2 + i * 40, MAP_HEIGHT // 2 + 50, 30, 30)
        for i in range(len(COLOR_OPTIONS))
    ]

    while True:
        screen.fill((50, 50, 50))
        
        # Title
        title = font_large.render("Karting Game Setup", True, (255, 255, 255))
        screen.blit(title, (MAP_WIDTH // 2 - title.get_width() // 2, 50))

        # Name input
        name_label = font.render("Name:", True, (255, 255, 255))
        screen.blit(name_label, (MAP_WIDTH // 2 - 150, MAP_HEIGHT // 2 - 50))
        name_surface = font.render(input_name + (cursor if input_active and cursor_visible else ''), True, (255, 255, 255))
        name_rect = pygame.Rect(MAP_WIDTH // 2 - 100, MAP_HEIGHT // 2 - 50, 200, 30)
        pygame.draw.rect(screen, (255, 255, 255), name_rect, 2)
        screen.blit(name_surface, (name_rect.x + 5, name_rect.y + 5))

        # Color selection
        color_label = font.render("Car Color:", True, (255, 255, 255))
        screen.blit(color_label, (MAP_WIDTH // 2 - 150, MAP_HEIGHT // 2))
        for i, button in enumerate(color_buttons):
            pygame.draw.rect(screen, COLOR_OPTIONS[i], button)
            if list(selected_color) == list(COLOR_OPTIONS[i]):
                pygame.draw.rect(screen, (255, 255, 255), button, 2)

        # Play button
        pygame.draw.rect(screen, (0, 200, 0), play_button)
        play_text = font.render("Play", True, (255, 255, 255))
        screen.blit(play_text, (play_button.x + (button_width - play_text.get_width()) // 2,
                               play_button.y + (button_height - play_text.get_height()) // 2))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
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

        # Cursor blink
        cursor_timer += 1
        if cursor_timer >= 30:
            cursor_visible = not cursor_visible
            cursor_timer = 0

def network_thread(local_car, player_name, player_color):
    global other_players, ping_times
    while True:
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
                'is_accelerating': local_car.is_accelerating,
                'is_braking': local_car.is_braking,
                'is_turning_left': local_car.is_turning_left,
                'is_turning_right': local_car.is_turning_right,
                'checkpoints_passed': local_car.checkpoints_passed
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
                            else:
                                car = Car(state['x'], state['y'], state['angle'], is_local_player=False)
                                car.velocity_x = state['velocity_x']
                                car.velocity_y = state['velocity_y']
                                car.speed = state['speed']
                                car.steering_angle = state['steering_angle']
                                car.angular_velocity = state['angular_velocity']
                            car.is_accelerating = state['is_accelerating']
                            car.is_braking = state['is_braking']
                            car.is_turning_left = state['is_turning_left']
                            car.is_turning_right = state['is_turning_right']
                            car.checkpoints_passed = state['checkpoints_passed']
                            car.color = color
                            car.name = name
                            other_players[pid] = {'car': car, 'last_update': current_time}
                    other_players_copy = other_players.copy()
                    for pid in other_players_copy:
                        if pid not in new_states and pid != PLAYER_ID:
                            del other_players[pid]
        except requests.RequestException as e:
            print(f"Network error: {e}")
        time.sleep(1/60)

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

    def update(self, keys=None):
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
            self.velocity_x *= -WALL_BOUNCE
            self.velocity_y *= -WALL_BOUNCE
            self.speed *= WALL_BOUNCE
            self.angular_velocity *= WALL_BOUNCE
        else:
            self.x = new_x
            self.y = new_y

        if self.render_enabled and not self.training_mode:
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

    def draw(self):
        if not self.render_enabled or self.training_mode:
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
        for x, y in points:
            rx = x * cos_angle - y * sin_angle
            ry = x * sin_angle + y * cos_angle
            rotated_points.append((self.x + rx, self.y + ry))
        pygame.draw.polygon(screen, self.color, rotated_points)

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
            for x, y in wheel_points:
                rx = x * cos_wheel - y * sin_wheel
                ry = x * sin_wheel + y * cos_wheel
                wheel_x = self.x + wx * cos_angle - wy * sin_angle
                wheel_y = self.y + wx * sin_angle + wy * cos_angle
                rotated_wheel.append((wheel_x + rx, wheel_y + ry))
            color = WHEEL_ACTIVE_COLOR if (i < 2 and (self.is_accelerating or self.is_braking)) or (i >= 2 and (self.is_turning_left or self.is_turning_right)) else WHEEL_COLOR
            pygame.draw.polygon(screen, color, rotated_wheel)

        render_text_with_outline(self.name, font, (255, 255, 255), (self.x - CAR_WIDTH // 2, self.y - CAR_HEIGHT - 80))
        render_text_with_outline(f"CP: {self.checkpoints_passed}", font, (0, 255, 0), (self.x - CAR_WIDTH // 2, self.y - CAR_HEIGHT - 60))

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
            impulse = dot / 2
            car1.velocity_x -= impulse * nx * CAR_COLLISION_BOUNCE
            car1.velocity_y -= impulse * ny * CAR_COLLISION_BOUNCE
            car2.velocity_x += impulse * nx * CAR_COLLISION_BOUNCE
            car2.velocity_y += impulse * ny * CAR_COLLISION_BOUNCE
            
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

# Start network thread
network_thread = threading.Thread(target=network_thread, args=(local_car, player_name, player_color), daemon=True)
network_thread.start()

def main():
    clock = pygame.time.Clock()
    FPS = 60
    last_time = time.time()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        keys = pygame.key.get_pressed()
        local_car.update(keys)

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

        screen.blit(map_image, (0, 0))
        screen.blit(trail_surface, (0, 0))
        
        with network_lock:
            for pid, data in other_players.items():
                if pid != PLAYER_ID:
                    data['car'].draw()
        
        local_car.draw()

        avg_ping = sum(ping_times) / len(ping_times) if ping_times else 0
        ping_text = f"Ping: {int(avg_ping)} ms"
        ping_surface = font.render(ping_text, True, (255, 255, 255))
        ping_pos = (MAP_WIDTH - ping_surface.get_width() - 10, 10)
        render_text_with_outline(ping_text, font, (255, 255, 255), ping_pos)

        pygame.display.flip()

        clock.tick(FPS)

if __name__ == "__main__":
    main()