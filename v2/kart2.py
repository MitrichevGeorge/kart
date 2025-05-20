import pygame
import sys
import math
import requests
import json
import threading
import time as tms
import uuid
import os
import random
from collections import deque
from particles import Explosion, DamagePopup, SmokeParticle, SparkParticle, NitroFlameParticle

pygame.init()

map_image = pygame.image.load('map.png')
MAP_WIDTH, MAP_HEIGHT = map_image.get_size()
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Karting Game")

MAP_SELECTION_MARGIN = 20
MAP_BUTTON_HEIGHT = 150
MAP_BUTTON_WIDTH = 300
MAP_THUMBNAIL_SIZE = (MAP_BUTTON_WIDTH - 40, MAP_BUTTON_HEIGHT - 60)
UPLOAD_BUTTON_HEIGHT = 50

# Карта
COLOR_FLOOR = (0, 0, 0)
COLOR_WALL = (200, 200, 200)
COLOR_SAND = (180, 180, 0)
COLOR_START = (50, 200, 0)

# Графика
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
ARROW_COLOR = (255, 255, 0, 128)
ARROW_LENGTH = 20
ARROW_THICKNESS = 3
ARROW_OFFSET = CAR_WIDTH
BUTTON_COLOR = (0, 200, 0)
BUTTON_HOVER_COLOR = (0, 150, 0)

# Физика
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
DRIFT_FACTOR_ON_SHIFT = 0.8
CAR_COLLISION_BOUNCE = 0.5
MIN_SPAWN_DISTANCE = 30
BLEND_FACTOR = 0.5
MAX_HEALTH = 20
DAMAGE_SCALING = 0.5
SPAWN_PROTECTION_TIME = 2.0
HEALTH_BAR_WIDTH = 40
HEALTH_BAR_HEIGHT = 6
HEALTH_BAR_OFFSET = 0
NITRO_BAR_OFFSET = 8
NAME_OFFSET = 30
SMOKE_HEALTH_THRESHOLD = 9
SMOKE_EMISSION_RATE = 0.1
SMOKE_LIFETIME = 1.0
SMOKE_SPEED = 10
POPUP_LIFETIME = 1.0
POPUP_SPEED = 20
EXPLOSION_LIFETIME = 0.5
EXPLOSION_SIZE = 40
CORPSE_LIFETIME = 3.0
SPARK_EMISSION_RATE = 0.1
SPARK_LIFETIME = 0.3
SPARK_SPEED = 15
SPARK_ALPHA_THRESHOLD = 50
NITRO_MAX = 100
NITRO_REGEN_RATE = 10
NITRO_CONSUMPTION_RATE = 50
NITRO_BOOST_FACTOR = 3.0
NITRO_LOW_THRESHOLD = 10
NITRO_LOW_SLOWDOWN = 0.7
NITRO_LOW_DAMAGE = 0.5
NITRO_FLAME_EMISSION_RATE = 0.05
NITRO_VISIBILITY_THRESHOLD = 0.95

trail_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)

font = pygame.font.SysFont('arial', 20)
font_large = pygame.font.SysFont('arial', 30)
font_small = pygame.font.SysFont('arial', 15)

SERVER_URL = 'http://geomit23.pythonanywhere.com/webhook'
PLAYER_ID = None  # Will be loaded or generated
other_players = {}
network_lock = threading.Lock()
ping_times = deque(maxlen=5)
connection_attempts = 0
MAX_CONNECTION_ATTEMPTS = 3
connection_established = False
is_paused = False
is_game_paused = False

COLOR_OPTIONS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 128)
]

checkpoints = {}
total_checkpoints = 0

lap_times = deque(maxlen=5)
current_lap_start = None
session_data = {}

def find_checkpoints():
    global checkpoints, total_checkpoints
    checkpoints = {}
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            color = map_image.get_at((x, y))[:3]
            if color[0] == 0 and color[2] == 0 and color[1] > 0:
                checkpoint_num = color[1]
                if checkpoint_num not in checkpoints:
                    checkpoints[checkpoint_num] = []
                checkpoints[checkpoint_num].append((x, y))
    total_checkpoints = len(checkpoints)
    sorted_checkpoints = {}
    for num in sorted(checkpoints.keys()):
        sorted_checkpoints[num] = checkpoints[num]
    checkpoints = sorted_checkpoints

find_checkpoints()

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
    global PLAYER_ID
    config_file = '.kart_config.json'
    default_config = {'name': 'Player', 'color': [128, 128, 128], 'session_id': str(uuid.uuid4()), 'position': None}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if (isinstance(config.get('name'), str) and
                    isinstance(config.get('color'), list) and
                    len(config['color']) == 3 and
                    is_valid_color(config['color'])):
                    PLAYER_ID = config.get('session_id', str(uuid.uuid4()))
                    return config
        except (json.JSONDecodeError, KeyError):
            pass
    PLAYER_ID = default_config['session_id']
    return default_config

def save_config(name, color, position=None):
    global PLAYER_ID
    config_file = '.kart_config.json'
    config = {'name': name, 'color': color, 'session_id': PLAYER_ID, 'position': position}
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f)
    except IOError as e:
        print(f"Failed to save config: {e}")

def load_session_data():
    global session_data
    session_file = '.kart_session.json'
    if os.path.exists(session_file):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            session_data = {'best_lap_times': {}, 'total_laps': {}}
    else:
        session_data = {'best_lap_times': {}, 'total_laps': {}}

def save_session_data():
    session_file = '.kart_session.json'
    try:
        with open(session_file, 'w') as f:
            json.dump(session_data, f)
    except IOError as e:
        print(f"Failed to save session data: {e}")

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

def draw_nitro_bar(screen, camera, x, y, nitro, max_nitro):
    nitro_ratio = max(0, min(1, nitro / max_nitro))
    bar_width = HEALTH_BAR_WIDTH * nitro_ratio
    color = (0, 191, 255)
    
    center_x, center_y = x - CAR_WIDTH // 2 + HEALTH_BAR_WIDTH // 2, y - CAR_HEIGHT - NITRO_BAR_OFFSET
    screen_pos = camera.apply_transform(None, (center_x, center_y))
    
    bg_rect = pygame.Rect(screen_pos[0] - HEALTH_BAR_WIDTH // 2, screen_pos[1] - HEALTH_BAR_HEIGHT // 2, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT)
    pygame.draw.rect(screen, (50, 50, 50), bg_rect, border_radius=HEALTH_BAR_HEIGHT // 2)
    
    if nitro_ratio > 0:
        nitro_rect = pygame.Rect(screen_pos[0] - HEALTH_BAR_WIDTH // 2, screen_pos[1] - HEALTH_BAR_HEIGHT // 2, bar_width, HEALTH_BAR_HEIGHT)
        pygame.draw.rect(screen, color, nitro_rect, border_radius=HEALTH_BAR_HEIGHT // 2)

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

        mouse_pos = pygame.mouse.get_pos()
        button_color = BUTTON_HOVER_COLOR if play_button.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(screen, button_color, play_button)
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

def show_connection_screen(attempt_number):
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT, connection_established, connection_attempts
    dots = ""
    start_time = tms.time()
    max_wait_time = 5.0
    while not connection_established and (tms.time() - start_time) < max_wait_time:
        screen.fill((50, 50, 50))
        
        elapsed = tms.time() - start_time
        dots = "." * (int(elapsed * 2) % 4)
        
        text = font_large.render(f"Connecting to server (Attempt {attempt_number}){dots}", True, (255, 255, 255))
        screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        tms.sleep(0.1)
    
    if not connection_established:
        connection_attempts = 0
        return False
    return True

def show_connection_lost_screen():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    text = font_large.render("Connection to server lost...", True, (255, 255, 255))
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

def show_pause_menu():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))
    screen.blit(overlay, (0, 0))

    button_width = 150
    button_height = 50
    continue_button = pygame.Rect(WINDOW_WIDTH // 2 - button_width // 2, WINDOW_HEIGHT // 2 - button_height * 2 - 20, button_width, button_height)
    laps_button = pygame.Rect(WINDOW_WIDTH // 2 - button_width // 2, WINDOW_HEIGHT // 2 - button_height // 2, button_width, button_height)
    times_button = pygame.Rect(WINDOW_WIDTH // 2 - button_width // 2, WINDOW_HEIGHT // 2 + button_height // 2 + 10, button_width, button_height)
    exit_button = pygame.Rect(WINDOW_WIDTH // 2 - button_width // 2, WINDOW_HEIGHT // 2 + button_height * 2 + 20, button_width, button_height)

    mouse_pos = pygame.mouse.get_pos()
    pygame.draw.rect(screen, BUTTON_HOVER_COLOR if continue_button.collidepoint(mouse_pos) else BUTTON_COLOR, continue_button)
    pygame.draw.rect(screen, BUTTON_HOVER_COLOR if laps_button.collidepoint(mouse_pos) else BUTTON_COLOR, laps_button)
    pygame.draw.rect(screen, BUTTON_HOVER_COLOR if times_button.collidepoint(mouse_pos) else BUTTON_COLOR, times_button)
    pygame.draw.rect(screen, BUTTON_HOVER_COLOR if exit_button.collidepoint(mouse_pos) else BUTTON_COLOR, exit_button)

    continue_text = font.render("Continue", True, (255, 255, 255))
    laps_text = font.render("Laps Leaderboard", True, (255, 255, 255))
    times_text = font.render("Times Leaderboard", True, (255, 255, 255))
    exit_text = font.render("Exit to Menu", True, (255, 255, 255))

    screen.blit(continue_text, (continue_button.x + (button_width - continue_text.get_width()) // 2,
                               continue_button.y + (button_height - continue_text.get_height()) // 2))
    screen.blit(laps_text, (laps_button.x + (button_width - laps_text.get_width()) // 2,
                           laps_button.y + (button_height - laps_text.get_height()) // 2))
    screen.blit(times_text, (times_button.x + (button_width - times_text.get_width()) // 2,
                            times_button.y + (button_height - times_text.get_height()) // 2))
    screen.blit(exit_text, (exit_button.x + (button_width - exit_text.get_width()) // 2,
                           exit_button.y + (button_height - exit_text.get_height()) // 2))

    return continue_button, laps_button, times_button, exit_button

def show_laps_leaderboard():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    screen.blit(overlay, (0, 0))

    leaderboard = []
    with network_lock:
        for pid, data in other_players.items():
            car = data['car']
            laps = car.lap_count - 1 + car.checkpoints_passed / total_checkpoints
            leaderboard.append((data['name'], laps))
        leaderboard.append((session_data['name'], (local_car.lap_count - 1 + local_car.checkpoints_passed / total_checkpoints)))

    leaderboard.sort(key=lambda x: x[1], reverse=True)
    
    title = font_large.render("Laps Leaderboard", True, (255, 255, 255))
    screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 50))

    for i, (name, laps) in enumerate(leaderboard[:10]):
        text = font.render(f"{i+1}. {name}: {laps:.2f} laps", True, (255, 255, 255))
        screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, 100 + i * 30))

    back_button = pygame.Rect(WINDOW_WIDTH // 2 - 75, WINDOW_HEIGHT - 100, 150, 50)
    mouse_pos = pygame.mouse.get_pos()
    pygame.draw.rect(screen, BUTTON_HOVER_COLOR if back_button.collidepoint(mouse_pos) else BUTTON_COLOR, back_button)
    back_text = font.render("Back", True, (255, 255, 255))
    screen.blit(back_text, (back_button.x + (150 - back_text.get_width()) // 2,
                           back_button.y + (50 - back_text.get_height()) // 2))

    return back_button

def show_times_leaderboard():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    screen.blit(overlay, (0, 0))

    leaderboard = []
    with network_lock:
        for pid, data in other_players.items():
            best_time = session_data['best_lap_times'].get(pid, float('inf'))
            if best_time != float('inf'):
                leaderboard.append((data['name'], best_time))
        my_best = session_data['best_lap_times'].get(PLAYER_ID, float('inf'))
        if my_best != float('inf'):
            leaderboard.append((session_data['name'], my_best))

    leaderboard.sort(key=lambda x: x[1])
    
    title = font_large.render("Best Lap Times", True, (255, 255, 255))
    screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 50))

    for i, (name, time) in enumerate(leaderboard[:10]):
        text = font.render(f"{i+1}. {name}: {time:.1f}s", True, (255, 255, 255))
        screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, 100 + i * 30))

    back_button = pygame.Rect(WINDOW_WIDTH // 2 - 75, WINDOW_HEIGHT - 100, 150, 50)
    mouse_pos = pygame.mouse.get_pos()
    pygame.draw.rect(screen, BUTTON_HOVER_COLOR if back_button.collidepoint(mouse_pos) else BUTTON_COLOR, back_button)
    back_text = font.render("Back", True, (255, 255, 255))
    screen.blit(back_text, (back_button.x + (150 - back_text.get_width()) // 2,
                           back_button.y + (50 - back_text.get_height()) // 2))

    return back_button

def network_thread(local_car, player_name, player_color):
    global other_players, ping_times, connection_attempts, connection_established, is_paused, is_game_paused
    while True:
        if (is_paused or is_game_paused) and connection_established:
            tms.sleep(0.1)
            continue
        try:
            start_time = tms.time()
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
                'lap_count': local_car.lap_count,
                'health': local_car.health,
                'is_dead': local_car.is_dead,
                'death_time': local_car.death_time if local_car.is_dead else 0,
                'nitro': local_car.nitro
            }
            payload = {
                'player_id': PLAYER_ID,
                'state': state,
                'name': player_name,
                'color': player_color
            }
            response = requests.post(SERVER_URL, json=payload, timeout=1)
            end_time = tms.time()
            
            ping_ms = (end_time - start_time) * 1000
            ping_times.append(ping_ms)
            
            if response.status_code == 200:
                connection_attempts = 0
                connection_established = True
                is_paused = False
                with network_lock:
                    new_states = response.json()
                    current_time = tms.time()
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
                                car.nitro = state.get('nitro', NITRO_MAX)
                                car.lap_count = state.get('lap_count', 1)
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
                                car.nitro = state.get('nitro', NITRO_MAX)
                                car.lap_count = state.get('lap_count', 1)
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
                    connection_established = False
                    is_paused = True
        except requests.RequestException:
            connection_attempts += 1
            if connection_attempts >= MAX_CONNECTION_ATTEMPTS:
                connection_established = False
                is_paused = True
        tms.sleep(1/30)

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
        self.is_drifting = False
        self.is_using_nitro = False
        self.checkpoints_passed = 0
        self.lap_count = 1  # Start at lap 1
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
        self.nitro = NITRO_MAX
        self.spawn_protection = True
        self.spawn_time = tms.time()
        self.spawn_x = x
        self.spawn_y = y
        self.damage_popups = []
        self.smoke_particles = []
        self.spark_particles = []
        self.nitro_flame_particles = []
        self.last_smoke_time = 0
        self.last_spark_time = 0
        self.last_nitro_flame_time = 0
        self.is_dead = False
        self.death_time = 0
        self.explosion = None

    def find_next_checkpoint(self):
        if not checkpoints:
            return None
        checkpoint_nums = sorted(checkpoints.keys())
        next_checkpoint_idx = self.checkpoints_passed % len(checkpoint_nums)
        return checkpoint_nums[next_checkpoint_idx]

    def find_nearest_checkpoint_pixel(self):
        next_checkpoint = self.find_next_checkpoint()
        if next_checkpoint is None:
            return None
        min_dist = float('inf')
        nearest_pos = None
        for pos in checkpoints[next_checkpoint]:
            dist = math.sqrt((self.x - pos[0])**2 + (self.y - pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_pos = pos
        return nearest_pos

    def check_checkpoint_collision(self):
        global current_lap_start, lap_times
        next_checkpoint = self.find_next_checkpoint()
        if next_checkpoint is None:
            return
        rect = pygame.Rect(self.x - CAR_WIDTH // 2, self.y - CAR_HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT)
        for pos in checkpoints[next_checkpoint]:
            if rect.collidepoint(pos):
                self.checkpoints_passed += 1
                if self.checkpoints_passed == total_checkpoints:
                    if self.is_local_player and current_lap_start is not None:
                        lap_time = tms.time() - current_lap_start
                        lap_times.append(lap_time)
                        session_data['best_lap_times'][PLAYER_ID] = min(session_data['best_lap_times'].get(PLAYER_ID, float('inf')), lap_time)
                        session_data['total_laps'][PLAYER_ID] = self.lap_count - 1 + self.checkpoints_passed / total_checkpoints
                        save_session_data()
                    self.lap_count += 1
                    self.checkpoints_passed = 0
                    if self.is_local_player:
                        current_lap_start = tms.time()
                elif self.is_local_player and self.checkpoints_passed == 1 and self.lap_count == 1:
                    current_lap_start = tms.time()
                break

    def update(self, keys=None):
        if self.is_dead and (tms.time() - self.death_time > CORPSE_LIFETIME):
            if self.is_local_player:
                self.is_dead = False
            else:
                return
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
            self.is_accelerating = keys[pygame.K_UP] or keys[pygame.K_w]
            self.is_braking = keys[pygame.K_DOWN] or keys[pygame.K_s]
            self.is_turning_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
            self.is_turning_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
            self.is_drifting = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            self.is_using_nitro = (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]) and (self.is_accelerating or self.is_braking)
        
        if self.is_accelerating:
            accel = ACCELERATION
        if self.is_braking:
            accel = -ACCELERATION * 0.5
        if self.is_using_nitro and self.nitro > 0:
            accel *= NITRO_BOOST_FACTOR
            self.nitro = max(0, self.nitro - NITRO_CONSUMPTION_RATE * delta_time)
        else:
            self.nitro = min(NITRO_MAX, self.nitro + NITRO_REGEN_RATE * delta_time)

        if self.nitro <= NITRO_LOW_THRESHOLD:
            self.speed *= NITRO_LOW_SLOWDOWN
            if not self.is_dead:
                damage = NITRO_LOW_DAMAGE * delta_time
                self.health = max(0, self.health - damage)
                if damage > 0.1:
                    self.damage_popups.append(DamagePopup(self.x, self.y, damage))

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

        drift_factor = DRIFT_FACTOR_ON_SHIFT if self.is_drifting else HIGH_SPEED_DRIFT_FACTOR
        drift_scale = 1 - speed_factor * drift_factor
        if speed_factor > 0.8 and abs(self.angular_velocity) > 0.01:
            self.angle += self.angular_velocity * speed_factor * 0.2

        direction_x = cos_angle
        direction_y = sin_angle
        current_drift_factor = 1 - speed_factor * drift_factor
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
            self.death_time = tms.time()
            self.explosion = Explosion(self.x, self.y)
            self.velocity_x = 0
            self.velocity_y = 0
            self.speed = 0
            self.angular_velocity = 0

        distance_from_spawn = math.sqrt((self.x - self.spawn_x)**2 + (self.y - self.spawn_y)**2)
        if (distance_from_spawn > MIN_SPAWN_DISTANCE or
            tms.time() - self.spawn_time > SPAWN_PROTECTION_TIME):
            self.spawn_protection = False

        if self.health <= SMOKE_HEALTH_THRESHOLD and not self.training_mode and not self.is_dead:
            current_time = tms.time()
            if current_time - self.last_smoke_time >= SMOKE_EMISSION_RATE:
                self.smoke_particles.append(SmokeParticle(self.x, self.y))
                self.last_smoke_time = current_time

        if self.is_using_nitro and self.nitro > 0 and not self.training_mode and not self.is_dead:
            current_time = tms.time()
            if current_time - self.last_nitro_flame_time >= NITRO_FLAME_EMISSION_RATE:
                for i, (wx, wy) in enumerate(self.wheel_positions):
                    if i < 2:
                        wheel_x = self.x + wx * cos_angle - wy * sin_angle
                        wheel_y = self.y + wx * sin_angle + wy * cos_angle
                        self.nitro_flame_particles.append(NitroFlameParticle(wheel_x, wheel_y, self.angle))
                self.last_nitro_flame_time = current_time

        self.check_checkpoint_collision()

        self.damage_popups = [popup for popup in self.damage_popups if popup.update(delta_time)]
        self.smoke_particles = [particle for particle in self.smoke_particles if particle.update(delta_time)]
        self.spark_particles = [particle for particle in self.spark_particles if particle.update(delta_time)]
        self.nitro_flame_particles = [particle for particle in self.nitro_flame_particles if particle.update(delta_time)]

        if self.render_enabled and not self.training_mode and not self.is_dead:
            self.draw_trails()

    def draw_trails(self):
        relative_speed = abs(self.speed) + abs(self.steering_angle * self.speed)
        trail_alpha = min(int(relative_speed / MAX_SPEED * 255 * FRICTION * 2), 255)
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        surface_color = get_surface_color(self.x, self.y)
        current_time = tms.time()

        for i, (wx, wy) in enumerate(self.wheel_positions):
            wheel_x = self.x + wx * cos_angle - wy * sin_angle
            wheel_y = self.y + wx * sin_angle + wy * cos_angle
            adjusted_trail_alpha = trail_alpha
            if i >= 2 and abs(self.steering_angle) > 0.01:
                adjusted_trail_alpha = min(trail_alpha * 1.5, 255)
            if adjusted_trail_alpha > 5:
                pygame.draw.circle(trail_surface, (*TRAIL_COLOR, adjusted_trail_alpha), (int(wheel_x), int(wheel_y)), 3)
                
            if (self.health <= SMOKE_HEALTH_THRESHOLD and not self.training_mode and not self.is_dead and
                surface_color == COLOR_FLOOR and adjusted_trail_alpha > SPARK_ALPHA_THRESHOLD and
                current_time - self.last_spark_time >= SPARK_EMISSION_RATE):
                self.spark_particles.append(SparkParticle(wheel_x, wheel_y))
                self.last_spark_time = current_time

    def draw(self, camera):
        if not self.render_enabled or self.training_mode:
            return
        if self.is_dead and (tms.time() - self.death_time > CORPSE_LIFETIME):
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

            nearest_checkpoint = self.find_nearest_checkpoint_pixel()
            if nearest_checkpoint:
                car_screen_pos = camera.apply_transform(None, (self.x, self.y))
                checkpoint_screen_pos = camera.apply_transform(None, nearest_checkpoint)
                dx = checkpoint_screen_pos[0] - car_screen_pos[0]
                dy = checkpoint_screen_pos[1] - car_screen_pos[1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist > 0:
                    norm_dx = dx / dist
                    norm_dy = dy / dist
                    # Offset the arrow start
                    arrow_start = (
                        car_screen_pos[0] + norm_dx * ARROW_OFFSET * camera.zoom,
                        car_screen_pos[1] + norm_dy * ARROW_OFFSET * camera.zoom
                    )
                    arrow_end = (
                        arrow_start[0] + norm_dx * ARROW_LENGTH * camera.zoom,
                        arrow_start[1] + norm_dy * ARROW_LENGTH * camera.zoom
                    )
                    # Create a surface for the arrow
                    arrow_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.line(arrow_surface, ARROW_COLOR, arrow_start, arrow_end, ARROW_THICKNESS)
                    # Draw arrowhead
                    arrowhead_angle = math.atan2(dy, dx)
                    arrowhead1 = (
                        arrow_end[0] - ARROW_LENGTH * 0.3 * camera.zoom * math.cos(arrowhead_angle + math.pi / 6),
                        arrow_end[1] - ARROW_LENGTH * 0.3 * camera.zoom * math.sin(arrowhead_angle + math.pi / 6)
                    )
                    arrowhead2 = (
                        arrow_end[0] - ARROW_LENGTH * 0.3 * camera.zoom * math.cos(arrowhead_angle - math.pi / 6),
                        arrow_end[1] - ARROW_LENGTH * 0.3 * camera.zoom * math.sin(arrowhead_angle - math.pi / 6)
                    )
                    pygame.draw.line(arrow_surface, ARROW_COLOR, arrow_end, arrowhead1, ARROW_THICKNESS)
                    pygame.draw.line(arrow_surface, ARROW_COLOR, arrow_end, arrowhead2, ARROW_THICKNESS)
                    screen.blit(arrow_surface, (0, 0))

        if self.explosion:
            self.explosion.draw(screen, camera)

        if not self.is_dead:
            render_text_with_outline(self.name, font, (255, 255, 255), (self.x - CAR_WIDTH // 2, self.y - CAR_HEIGHT - NAME_OFFSET), camera)
            draw_health_bar(screen, camera, self.x, self.y, self.health, MAX_HEALTH)
            if self.nitro < NITRO_MAX * NITRO_VISIBILITY_THRESHOLD:
                draw_nitro_bar(screen, camera, self.x, self.y, self.nitro, NITRO_MAX)

        for popup in self.damage_popups:
            popup.draw(screen, camera)
        for particle in self.smoke_particles:
            particle.draw(screen, camera)
        for particle in self.spark_particles:
            particle.draw(screen, camera)
        for particle in self.nitro_flame_particles:
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
        self.lap_count = 1
        self.total_reward = 0
        self.health = MAX_HEALTH
        self.nitro = NITRO_MAX
        self.spawn_protection = True
        self.spawn_time = tms.time()
        self.spawn_x = x
        self.spawn_y = y
        self.damage_popups = []
        self.smoke_particles = []
        self.spark_particles = []
        self.nitro_flame_particles = []
        self.last_smoke_time = 0
        self.last_spark_time = 0
        self.last_nitro_flame_time = 0
        self.is_dead = False
        self.death_time = 0
        self.explosion = None

def get_surface_color(x, y):
    if 0 <= x < MAP_WIDTH and 0 <= y < MAP_HEIGHT:
        return map_image.get_at((int(x), int(y)))[:3]
    return COLOR_WALL

def find_start_position():
    config = load_config()
    if config.get('position') and get_surface_color(*config['position']) != COLOR_WALL:
        return config['position']
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
            car1.nitro = max(0, car1.nitro - damage)
            car2.nitro = max(0, car2.nitro - damage)
            
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

def attempt_game_start(player_name, player_color):
    global connection_established, is_game_paused, session_data
    load_session_data()
    session_data['name'] = player_name
    start_x, start_y = find_start_position()
    local_car = Car(start_x, start_y, 0, is_local_player=True)
    local_car.name = player_name
    local_car.color = player_color
    camera = Camera()
    camera.x = start_x
    camera.y = start_y
    connection_established = False
    is_game_paused = False
    network_thread_obj = threading.Thread(target=network_thread, args=(local_car, player_name, player_color), daemon=True)
    network_thread_obj.start()
    tms.sleep(0.5)
    return local_car, camera, network_thread_obj

def main(local_car, camera):
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT, is_paused, is_game_paused, connection_established, current_lap_start
    clock = pygame.time.Clock()
    FPS = 60
    last_time = tms.time()  # Use tms.time()
    show_laps_leaderboard_flag = False
    show_times_leaderboard_flag = False
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_config(local_car.name, local_car.color, [local_car.x, local_car.y])
                save_session_data()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEWHEEL:
                camera.adjust_zoom(event.y * 0.1)
            elif event.type == pygame.VIDEORESIZE:
                WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if local_car.is_dead:
                    start_x, start_y = find_start_position()
                    local_car.reset(start_x, start_y)
                    camera.x = start_x
                    camera.y = start_y
                elif is_game_paused:
                    continue_button, laps_button, times_button, exit_button = show_pause_menu()
                    if continue_button.collidepoint(event.pos):
                        is_game_paused = False
                        show_laps_leaderboard_flag = False
                        show_times_leaderboard_flag = False
                    elif laps_button.collidepoint(event.pos):
                        show_laps_leaderboard_flag = True
                        show_times_leaderboard_flag = False
                    elif times_button.collidepoint(event.pos):
                        show_times_leaderboard_flag = True
                        show_laps_leaderboard_flag = False
                    elif exit_button.collidepoint(event.pos):
                        save_config(local_car.name, local_car.color, [local_car.x, local_car.y])
                        save_session_data()
                        return False
                    elif show_laps_leaderboard_flag:
                        back_button = show_laps_leaderboard()
                        if back_button.collidepoint(event.pos):
                            show_laps_leaderboard_flag = False
                    elif show_times_leaderboard_flag:
                        back_button = show_times_leaderboard()
                        if back_button.collidepoint(event.pos):
                            show_times_leaderboard_flag = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    is_game_paused = not is_game_paused
                    show_laps_leaderboard_flag = False
                    show_times_leaderboard_flag = False

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
            checkpoint_text = f"Checkpoints: {local_car.checkpoints_passed} / {total_checkpoints}"
            lap_text = f"Lap: {local_car.lap_count}"
            render_text_with_outline(checkpoint_text, font, (255, 255, 255), (WINDOW_WIDTH // 2, 10), camera=None)
            render_text_with_outline(lap_text, font, (255, 255, 255), (WINDOW_WIDTH // 2, 30), camera=None)
            pygame.display.flip()
            clock.tick(FPS)
            connection_attempts = 0
            connection_established = False
            if not show_connection_screen(1):
                return False
            continue

        if is_game_paused:
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
            if show_laps_leaderboard_flag:
                show_laps_leaderboard()
            elif show_times_leaderboard_flag:
                show_times_leaderboard()
            else:
                show_pause_menu()
            checkpoint_text = f"Checkpoints: {local_car.checkpoints_passed} / {total_checkpoints}"
            lap_text = f"Lap: {local_car.lap_count}"
            render_text_with_outline(checkpoint_text, font, (255, 255, 255), (WINDOW_WIDTH // 2, 10), camera=None)
            render_text_with_outline(lap_text, font, (255, 255, 255), (WINDOW_WIDTH // 2, 30), camera=None)
            pygame.display.flip()
            clock.tick(FPS)
            continue

        current_time = tms.time()
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

        # Draw current lap time and last 5 lap times
        if current_lap_start is not None:
            lap_time = tms.time() - current_lap_start
            lap_time_text = f"Current Lap: {lap_time:.1f}s"
            render_text_with_outline(lap_time_text, font, (255, 255, 255), (10, 10), camera=None)
            for i, time_val in enumerate(lap_times):
                lap_text = f"Lap {len(lap_times) - i}: {time_val:.1f}s"
                render_text_with_outline(lap_text, font_small, (255, 255, 255), (10, 40 + i * 20), camera=None)

        avg_ping = sum(ping_times) / len(ping_times) if ping_times else 0
        ping_text = f"Ping: {int(avg_ping)} ms"
        ping_surface = font.render(ping_text, True, (255, 255, 255))
        ping_pos = (WINDOW_WIDTH - ping_surface.get_width() - 10, 10)
        render_text_with_outline(ping_text, font, (255, 255, 255), ping_pos, camera=None)

        checkpoint_text = f"Checkpoints: {local_car.checkpoints_passed} / {total_checkpoints}"
        lap_text = f"Lap: {local_car.lap_count}"
        render_text_with_outline(checkpoint_text, font, (255, 255, 255), (WINDOW_WIDTH // 2, 10), camera=None)
        render_text_with_outline(lap_text, font, (255, 255, 255), (WINDOW_WIDTH // 2, 30), camera=None)

        pygame.display.flip()
        clock.tick(FPS)

def show_map_upload_screen():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    
    uploaded = False
    map_file = None
    map_name = ""
    input_active = False
    cursor = '_'
    cursor_timer = 0
    cursor_visible = True
    
    # Кнопки
    select_button = pygame.Rect(WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 50, 300, 50)
    upload_button = pygame.Rect(WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 + 50, 300, 50)
    back_button = pygame.Rect(20, WINDOW_HEIGHT - 70, 100, 50)
    
    # Поле для имени карты
    name_rect = pygame.Rect(WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 150, 300, 40)
    
    while True:
        screen.fill((50, 50, 50))
        
        # Заголовок
        title = font_large.render("Upload Custom Map", True, (255, 255, 255))
        screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 50))
        
        # Поле для имени карты
        pygame.draw.rect(screen, (255, 255, 255), name_rect, 2)
        name_label = font.render("Map Name:", True, (255, 255, 255))
        screen.blit(name_label, (name_rect.x, name_rect.y - 30))
        name_surface = font.render(map_name + (cursor if input_active and cursor_visible else ''), True, (255, 255, 255))
        screen.blit(name_surface, (name_rect.x + 10, name_rect.y + 10))
        
        # Кнопка выбора файла
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.rect(screen, BUTTON_HOVER_COLOR if select_button.collidepoint(mouse_pos) else BUTTON_COLOR, select_button)
        select_text = font.render("Select Map File (PNG)", True, (255, 255, 255))
        screen.blit(select_text, (
            select_button.x + (select_button.width - select_text.get_width()) // 2,
            select_button.y + (select_button.height - select_text.get_height()) // 2
        ))
        
        # Кнопка загрузки
        upload_enabled = map_file is not None and map_name.strip() != ""
        upload_color = BUTTON_HOVER_COLOR if upload_button.collidepoint(mouse_pos) and upload_enabled else (
            (100, 100, 100) if not upload_enabled else BUTTON_COLOR
        )
        pygame.draw.rect(screen, upload_color, upload_button)
        upload_text = font.render("Upload Map", True, (255, 255, 255))
        screen.blit(upload_text, (
            upload_button.x + (upload_button.width - upload_text.get_width()) // 2,
            upload_button.y + (upload_button.height - upload_text.get_height()) // 2
        ))
        
        # Кнопка назад
        pygame.draw.rect(screen, BUTTON_HOVER_COLOR if back_button.collidepoint(mouse_pos) else BUTTON_COLOR, back_button)
        back_text = font.render("Back", True, (255, 255, 255))
        screen.blit(back_text, (
            back_button.x + (back_button.width - back_text.get_width()) // 2,
            back_button.y + (back_button.height - back_text.get_height()) // 2
        ))
        
        # Информация о выбранном файле
        if map_file:
            file_text = font.render(f"Selected: {map_file}", True, (200, 200, 200))
            screen.blit(file_text, (select_button.x, select_button.y + select_button.height + 10))
        
        # Сообщение об успешной загрузке
        if uploaded:
            success_text = font.render("Map uploaded successfully!", True, (0, 255, 0))
            screen.blit(success_text, (WINDOW_WIDTH // 2 - success_text.get_width() // 2, upload_button.y + upload_button.height + 20))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if back_button.collidepoint(event.pos):
                    return False
                elif select_button.collidepoint(event.pos):
                    # Открываем диалог выбора файла
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                        root = tk.Tk()
                        root.withdraw()
                        file_path = filedialog.askopenfilename(
                            title="Select Map Image",
                            filetypes=[("PNG files", "*.png")]
                        )
                        if file_path:
                            map_file = file_path
                    except:
                        pass
                elif upload_button.collidepoint(event.pos) and upload_enabled:
                    # Загружаем карту на сервер
                    try:
                        with open(map_file, 'rb') as f:
                            files = {'file': f}
                            data = {
                                'name': map_name,
                                'physics': json.dumps(DEFAULT_PHYSICS)
                            }
                            response = requests.post(
                                f'{SERVER_URL}/maps/upload',
                                files=files,
                                data=data
                            )
                            if response.status_code == 200:
                                uploaded = True
                                map_file = None
                                map_name = ""
                    except:
                        pass
                elif name_rect.collidepoint(event.pos):
                    input_active = True
                else:
                    input_active = False
            elif event.type == pygame.KEYDOWN and input_active:
                if event.key == pygame.K_BACKSPACE:
                    map_name = map_name[:-1]
                elif event.key == pygame.K_RETURN:
                    input_active = False
                elif event.unicode.isalnum() or event.unicode in [' ', '-', '_']:
                    if len(map_name) < 20:
                        map_name += event.unicode
        
        # Мигание курсора
        cursor_timer += 1
        if cursor_timer >= 30:
            cursor_visible = not cursor_visible
            cursor_timer = 0

def show_map_selection_screen():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    
    try:
        response = requests.get(f'{SERVER_URL}/maps')
        if response.status_code == 200:
            available_maps = response.json()
        else:
            available_maps = {}
    except requests.RequestException:
        available_maps = {}
    
    map_buttons = []
    upload_button = pygame.Rect(
        WINDOW_WIDTH // 2 - 100,
        WINDOW_HEIGHT - UPLOAD_BUTTON_HEIGHT - 20,
        200,
        UPLOAD_BUTTON_HEIGHT
    )
    
    while True:
        screen.fill((50, 50, 50))
        
        title = font_large.render("Select a Map", True, (255, 255, 255))
        screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 20))
        
        # Draw map buttons
        map_buttons = []
        for i, (map_id, map_data) in enumerate(available_maps.items()):
            row = i // 2
            col = i % 2
            x = MAP_SELECTION_MARGIN + col * (MAP_BUTTON_WIDTH + MAP_SELECTION_MARGIN)
            y = 80 + row * (MAP_BUTTON_HEIGHT + MAP_SELECTION_MARGIN)
            rect = pygame.Rect(x, y, MAP_BUTTON_WIDTH, MAP_BUTTON_HEIGHT)
            map_buttons.append((rect, map_id))
            
            # Draw button background
            pygame.draw.rect(screen, (100, 100, 100), rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 2)
            
            # Try to load thumbnail
            try:
                response = requests.get(f'{SERVER_URL}/maps/{map_id}')
                if response.status_code == 200:
                    with open('temp_map.png', 'wb') as f:
                        f.write(response.content)
                    thumbnail = pygame.image.load('temp_map.png')
                    thumbnail = pygame.transform.scale(thumbnail, MAP_THUMBNAIL_SIZE)
                    screen.blit(thumbnail, (x + 20, y + 40))
                    os.remove('temp_map.png')
            except:
                pass
            
            # Draw map name
            name_text = font.render(map_data['name'], True, (255, 255, 255))
            screen.blit(name_text, (x + 10, y + 10))
        
        # Draw upload button
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.rect(screen, BUTTON_HOVER_COLOR if upload_button.collidepoint(mouse_pos) else BUTTON_COLOR, upload_button)
        upload_text = font.render("Add Map", True, (255, 255, 255))
        screen.blit(upload_text, (
            upload_button.x + (upload_button.width - upload_text.get_width()) // 2,
            upload_button.y + (upload_button.height - upload_text.get_height()) // 2
        ))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if upload_button.collidepoint(event.pos):
                    show_map_upload_screen()
                    # После закрытия экрана загрузки обновляем список карт
                    try:
                        response = requests.get(f'{SERVER_URL}/maps')
                        if response.status_code == 200:
                            available_maps = response.json()
                    except requests.RequestException:
                        pass
                for rect, map_id in map_buttons:
                    if rect.collidepoint(event.pos):
                        return map_id

# Modify the main game loop
if __name__ == "__main__":
    while True:
        player_name, player_color = show_start_screen()
        
        # Show map selection screen
        selected_map = show_map_selection_screen()
        if selected_map == "upload":
            # Implement map upload functionality
            pass
        else:
            # Load selected map and its physics
            try:
                response = requests.get(f'{SERVER_URL}/maps/{selected_map}')
                if response.status_code == 200:
                    with open('current_map.png', 'wb') as f:
                        f.write(response.content)
                    map_image = pygame.image.load('current_map.png')
                    MAP_WIDTH, MAP_HEIGHT = map_image.get_size()
                    
                    # Get physics for this map
                    response = requests.get(f'{SERVER_URL}/maps')
                    if response.status_code == 200:
                        maps_data = response.json()
                        if selected_map in maps_data:
                            physics = maps_data[selected_map]['physics']
                            # Update physics constants
                            globals().update(physics)
            except requests.RequestException:
                pass
            
            # Rest of the game initialization
            find_checkpoints()
            
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                local_car, camera, network_thread_obj = attempt_game_start(player_name, player_color)
                if show_connection_screen(attempt):
                    if main(local_car, camera):
                        break
                if attempt < max_attempts:
                    other_players.clear()
                    ping_times.clear()
                    connection_attempts = 0
                    connection_established = False
            else:
                continue