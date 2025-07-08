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
from particles import Explosion, DamagePopup, SmokeParticle, SparkParticle, NitroFlameParticle
from io import BytesIO
from camera import Camera

screen = None
font = None
font_large = None
font_small = None
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600

COLOR_FLOOR = (0, 0, 0)
COLOR_WALL = (200, 200, 200)
COLOR_SAND = (180, 180, 0)
COLOR_START = (50, 200, 0)
COLOR_SPRING_WALL = (0, 200, 50)

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

PHYSICS_PARAMS = {}
trail_surface = None
map_image = None
MAP_WIDTH = 0
MAP_HEIGHT = 0

BUTTON_COLOR = (0, 200, 0)
BUTTON_HOVER_COLOR = (0, 150, 0)

SERVER_URL = None
PLAYER_ID = None
other_players = {}
network_lock = threading.Lock()
ping_times = deque(maxlen=5)
connection_attempts = 0
MAX_CONNECTION_ATTEMPTS = 3
connection_established = False
is_paused = False
is_game_paused = False

checkpoints = {}
total_checkpoints = 0
lap_times = deque(maxlen=5)
current_lap_start = None

def setup_screen(sq):
    global screen, font, font_large, font_small
    screen = sq
    font = pygame.font.SysFont('arial', 20)
    font_large = pygame.font.SysFont('arial', 30)
    font_small = pygame.font.SysFont('arial', 15)

def load_map_and_params(server_url):
    global map_image, MAP_WIDTH, MAP_HEIGHT, trail_surface, PHYSICS_PARAMS
    try:
        map_response = requests.get(f'http://{server_url}/map', timeout=5)
        if map_response.status_code == 200:
            map_data = BytesIO(map_response.content)
            map_image = pygame.image.load(map_data)
            MAP_WIDTH, MAP_HEIGHT = map_image.get_size()
            trail_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)
        else:
            raise Exception("Failed to load map from server")
        info_response = requests.get(f'http://{server_url}/info', timeout=5)
        if info_response.status_code == 200:
            PHYSICS_PARAMS.update(info_response.json())
        else:
            raise Exception("Failed to load physics parameters from server")
    except Exception as e:
        print(f"Error loading map or params: {e}")
        return False
    return True

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
    sorted_checkpoints = {num: checkpoints[num] for num in sorted(checkpoints.keys())}
    checkpoints = sorted_checkpoints

def is_valid_color(color):
    r, g, b = color
    return r + g + b >= 150 and max(r, g, b) >= 50

def load_config():
    global PLAYER_ID
    config_file = '.kart_config.json'
    default_config = {'name': 'Player', 'color': [128, 128, 128], 'session_id': str(uuid.uuid4()), 'position': None, 'server_url': 'geomit25'}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if (isinstance(config.get('name'), str) and
                    isinstance(config.get('color'), list) and
                    len(config['color']) == 3 and
                    is_valid_color(config['color']) and
                    isinstance(config.get('server_url'), str)):
                    PLAYER_ID = config.get('session_id', str(uuid.uuid4()))
                    return config
        except (json.JSONDecodeError, KeyError):
            pass
    PLAYER_ID = default_config['session_id']
    return default_config

def save_config(name, color, server_url, position=None):
    global PLAYER_ID
    config_file = '.kart_config.json'
    config = {'name': name, 'color': color, 'session_id': PLAYER_ID, 'server_url': server_url, 'position': position}
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f)
    except IOError as e:
        print(f"Failed to save config: {e}")

def load_session_data():
    global session_data
    session_file = '.kart_session.json'
    session_data = {'best_lap_times': {}, 'total_laps': {}}
    if os.path.exists(session_file):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

def save_session_data():
    session_file = '.kart_session.json'
    try:
        with open(session_file, 'w') as f:
            json.dump(session_data, f)
    except IOError as e:
        print(f"Failed to save session data: {e}")

def draw_health_bar(screen, camera, x, y, health, max_health):
    health_ratio = max(0, min(1, health / max_health))
    bar_width = PHYSICS_PARAMS['HEALTH_BAR_WIDTH'] * health_ratio
    green = int(255 * health_ratio)
    red = int(255 * (1 - health_ratio))
    color = (red, green, 0)
    center_x = x - CAR_WIDTH // 2 + PHYSICS_PARAMS['HEALTH_BAR_WIDTH'] // 2
    center_y = y - CAR_HEIGHT - PHYSICS_PARAMS['HEALTH_BAR_OFFSET']
    screen_pos = camera.apply_transform(None, (center_x, center_y))
    bg_rect = pygame.Rect(
        screen_pos[0] - PHYSICS_PARAMS['HEALTH_BAR_WIDTH'] // 2,
        screen_pos[1] - PHYSICS_PARAMS['HEALTH_BAR_HEIGHT'] // 2,
        PHYSICS_PARAMS['HEALTH_BAR_WIDTH'],
        PHYSICS_PARAMS['HEALTH_BAR_HEIGHT']
    )
    pygame.draw.rect(screen, (50, 50, 50), bg_rect, border_radius=PHYSICS_PARAMS['HEALTH_BAR_HEIGHT'] // 2)
    if health_ratio > 0:
        health_rect = pygame.Rect(
            screen_pos[0] - PHYSICS_PARAMS['HEALTH_BAR_WIDTH'] // 2,
            screen_pos[1] - PHYSICS_PARAMS['HEALTH_BAR_HEIGHT'] // 2,
            bar_width,
            PHYSICS_PARAMS['HEALTH_BAR_HEIGHT']
        )
        pygame.draw.rect(screen, color, health_rect, border_radius=PHYSICS_PARAMS['HEALTH_BAR_HEIGHT'] // 2)

def draw_nitro_bar(screen, camera, x, y, nitro, max_nitro):
    nitro_ratio = max(0, min(1, nitro / max_nitro))
    bar_width = PHYSICS_PARAMS['HEALTH_BAR_WIDTH'] * nitro_ratio
    color = (0, 191, 255)
    center_x = x - CAR_WIDTH // 2 + PHYSICS_PARAMS['HEALTH_BAR_WIDTH'] // 2
    center_y = y - CAR_HEIGHT - PHYSICS_PARAMS['NITRO_BAR_OFFSET']
    screen_pos = camera.apply_transform(None, (center_x, center_y))
    bg_rect = pygame.Rect(
        screen_pos[0] - PHYSICS_PARAMS['HEALTH_BAR_WIDTH'] // 2,
        screen_pos[1] - PHYSICS_PARAMS['HEALTH_BAR_HEIGHT'] // 2,
        PHYSICS_PARAMS['HEALTH_BAR_WIDTH'],
        PHYSICS_PARAMS['HEALTH_BAR_HEIGHT']
    )
    pygame.draw.rect(screen, (50, 50, 50), bg_rect, border_radius=PHYSICS_PARAMS['HEALTH_BAR_HEIGHT'] // 2)
    if nitro_ratio > 0:
        nitro_rect = pygame.Rect(
            screen_pos[0] - PHYSICS_PARAMS['HEALTH_BAR_WIDTH'] // 2,
            screen_pos[1] - PHYSICS_PARAMS['HEALTH_BAR_HEIGHT'] // 2,
            bar_width,
            PHYSICS_PARAMS['HEALTH_BAR_HEIGHT']
        )
        pygame.draw.rect(screen, color, nitro_rect, border_radius=PHYSICS_PARAMS['HEALTH_BAR_HEIGHT'] // 2)

def render_text_with_outline(text, font, color, pos, camera=None):
    inv_color = (255 - color[0], 255 - color[1], 255 - color[2])
    text_surface = font.render(text, True, color)
    outline_surface = font.render(text, True, inv_color)
    screen_pos = camera.apply_transform(None, pos) if camera else pos
    outline_positions = [
        (screen_pos[0] - 1, screen_pos[1]),
        (screen_pos[0] + 1, screen_pos[1]),
        (screen_pos[0], screen_pos[1] - 1),
        (screen_pos[0], screen_pos[1] + 1)
    ]
    for outline_pos in outline_positions:
        screen.blit(outline_surface, outline_pos)
    screen.blit(text_surface, screen_pos)

def show_connection_screen(attempt_number):
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT, connection_established, connection_attempts
    dots = ""
    start_time = time.time()
    max_wait_time = 5.0
    while not connection_established and (time.time() - start_time) < max_wait_time:
        screen.fill((50, 50, 50))
        elapsed = time.time() - start_time
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
        time.sleep(0.1)
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
            response = requests.post(f'http://{SERVER_URL}/webhook', json=payload, timeout=1)
            ping_times.append((time.time() - start_time) * 1000)
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
                                car.x = PHYSICS_PARAMS['BLEND_FACTOR'] * car.x + (1 - PHYSICS_PARAMS['BLEND_FACTOR']) * state['x']
                                car.y = PHYSICS_PARAMS['BLEND_FACTOR'] * car.y + (1 - PHYSICS_PARAMS['BLEND_FACTOR']) * state['y']
                                car.angle = PHYSICS_PARAMS['BLEND_FACTOR'] * car.angle + (1 - PHYSICS_PARAMS['BLEND_FACTOR']) * state['angle']
                                car.speed = PHYSICS_PARAMS['BLEND_FACTOR'] * car.speed + (1 - PHYSICS_PARAMS['BLEND_FACTOR']) * state['speed']
                                car.steering_angle = PHYSICS_PARAMS['BLEND_FACTOR'] * car.steering_angle + (1 - PHYSICS_PARAMS['BLEND_FACTOR']) * state['steering_angle']
                                car.velocity_x = PHYSICS_PARAMS['BLEND_FACTOR'] * car.velocity_x + (1 - PHYSICS_PARAMS['BLEND_FACTOR']) * state['velocity_x']
                                car.velocity_y = PHYSICS_PARAMS['BLEND_FACTOR'] * car.velocity_y + (1 - PHYSICS_PARAMS['BLEND_FACTOR']) * state['velocity_y']
                                car.angular_velocity = PHYSICS_PARAMS['BLEND_FACTOR'] * car.angular_velocity + (1 - PHYSICS_PARAMS['BLEND_FACTOR']) * state['angular_velocity']
                                car.health = state.get('health', PHYSICS_PARAMS['MAX_HEALTH'])
                                car.is_dead = state.get('is_dead', False)
                                car.death_time = state.get('death_time', 0)
                                car.nitro = state.get('nitro', PHYSICS_PARAMS['NITRO_MAX'])
                                car.lap_count = state.get('lap_count', 1)
                            else:
                                car = Car(state['x'], state['y'], state['angle'], is_local_player=False)
                                car.velocity_x = state['velocity_x']
                                car.velocity_y = state['velocity_y']
                                car.speed = state['speed']
                                car.steering_angle = state['steering_angle']
                                car.angular_velocity = state['angular_velocity']
                                car.health = state.get('health', PHYSICS_PARAMS['MAX_HEALTH'])
                                car.is_dead = state.get('is_dead', False)
                                car.death_time = state.get('death_time', 0)
                                car.nitro = state.get('nitro', PHYSICS_PARAMS['NITRO_MAX'])
                                car.lap_count = state.get('lap_count', 1)
                            car.checkpoints_passed = state['checkpoints_passed']
                            car.color = color
                            car.name = name
                            other_players[pid] = {'car': car, 'last_update': current_time}
                    for pid in list(other_players.keys()):
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
        time.sleep(1/30)



def get_surface_color(x, y):
    if map_image is None or x < 0 or x >= MAP_WIDTH or y < 0 or y >= MAP_HEIGHT:
        return COLOR_WALL
    return map_image.get_at((int(x), int(y)))[:3]

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
            car1.velocity_x -= impulse * nx * PHYSICS_PARAMS['CAR_COLLISION_BOUNCE']
            car1.velocity_y -= impulse * ny * PHYSICS_PARAMS['CAR_COLLISION_BOUNCE']
            car2.velocity_x += impulse * nx * PHYSICS_PARAMS['CAR_COLLISION_BOUNCE']
            car2.velocity_y += impulse * ny * PHYSICS_PARAMS['CAR_COLLISION_BOUNCE']
            damage = impulse_magnitude * PHYSICS_PARAMS['DAMAGE_SCALING']
            if damage > 0.1:
                car1.damage_popups.append(DamagePopup(car1.x, car1.y, damage))
                car2.damage_popups.append(DamagePopup(car2.x, car2.y, damage))
            car1.health = max(0, car1.health - damage)
            car2.health = max(0, car2.health - damage)
            car1.nitro = max(0, car1.nitro - damage)
            car2.nitro = max(0, car2.nitro - damage)
            car1.speed *= PHYSICS_PARAMS['CAR_COLLISION_BOUNCE']
            car2.speed *= PHYSICS_PARAMS['CAR_COLLISION_BOUNCE']
            car1.angular_velocity *= PHYSICS_PARAMS['CAR_COLLISION_BOUNCE']
            car2.angular_velocity *= PHYSICS_PARAMS['CAR_COLLISION_BOUNCE']
            overlap = (CAR_WIDTH + CAR_HEIGHT) / 2 - distance
            if overlap > 0:
                car1.x += nx * overlap / 2
                car1.y += ny * overlap / 2
                car2.x -= nx * overlap / 2
                car2.y -= ny * overlap / 2

def attempt_game_start(player_name, player_color, server_url):
    global connection_established, is_game_paused, session_data, SERVER_URL, map_image
    SERVER_URL = server_url
    load_session_data()
    session_data['name'] = player_name
    pygame.display.set_caption(f"Karting Game - {player_name} - {server_url}")
    if not load_map_and_params(server_url):
        return None, None, None
    find_checkpoints()
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
    time.sleep(0.5)
    return local_car, camera, network_thread_obj

def main(local_car, camera):
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT, is_paused, is_game_paused, connection_established, current_lap_start
    clock = pygame.time.Clock()
    FPS = 60
    last_time = time.time()
    show_laps_leaderboard_flag = False
    show_times_leaderboard_flag = False
    last_zoom = -1
    scaled_map = None
    scaled_trails = None
    fade_counter = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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
        if last_zoom != camera.zoom:
            scaled_map = camera.apply_surface_transform(map_image, (0, 0))
            scaled_trails = camera.apply_surface_transform(trail_surface, (0, 0))
            last_zoom = camera.zoom
        map_pos = camera.getpos((0, 0))
        screen.fill((0, 0, 0))
        screen.blit(scaled_map, map_pos)
        screen.blit(scaled_trails, map_pos)
        if is_paused:
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
        fade_counter += 1
        if fade_counter >= 10:
            faded_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)
            faded_surface.blit(trail_surface, (0, 0))
            faded_surface.set_alpha(int(255 * PHYSICS_PARAMS['TRAIL_FADE_RATE'] * 10))
            trail_surface.fill((0, 0, 0, 0))
            trail_surface.blit(faded_surface, (0, 0))
            fade_counter = 0
        with network_lock:
            for pid, data in other_players.items():
                if pid != PLAYER_ID:
                    data['car'].draw(camera)
        local_car.draw(camera)
        if local_car.is_dead:
            show_death_screen()
        if current_lap_start is not None:
            lap_time = time.time() - current_lap_start
            lap_time_text = f"Current Lap: {lap_time:.1f}s"
            render_text_with_outline(lap_time_text, font, (255, 255, 255), (10, 10), camera=None)
            for i, time_val in enumerate(lap_times):
                lap_text = f"Lap {len(lap_times) - i}: {time_val:.1f}s"
                render_text_with_outline(lap_text, font_small, (255, 255, 255), (10, 40 + i * 20), camera=None)
        avg_ping = sum(ping_times) / len(ping_times) if ping_times else 0
        ping_text = f"Ping: {int(avg_ping)} ms"
        render_text_with_outline(ping_text, font, (255, 255, 255), (WINDOW_WIDTH - font.render(ping_text, True, (255, 255, 255)).get_width() - 10, 10), camera=None)
        fps_text = f"FPS: {int(clock.get_fps())}"
        render_text_with_outline(fps_text, font, (255, 255, 255), (WINDOW_WIDTH - font.render(fps_text, True, (255, 255, 255)).get_width() - 10, 30), camera=None)
        checkpoint_text = f"Checkpoints: {local_car.checkpoints_passed} / {total_checkpoints}"
        lap_text = f"Lap: {local_car.lap_count}"
        render_text_with_outline(checkpoint_text, font, (255, 255, 255), (WINDOW_WIDTH // 2, 10), camera=None)
        render_text_with_outline(lap_text, font, (255, 255, 255), (WINDOW_WIDTH // 2, 30), camera=None)
        pygame.display.flip()
        clock.tick(FPS)