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
from io import BytesIO
from game import *
import threading
import time
import urllib.request
from tk import parse_description, wrap_text, render_lines

pygame.init()

WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Karting Game")

setup_screen(screen)

BUTTON_COLOR = (0, 200, 0)
BUTTON_HOVER_COLOR = (0, 150, 0)
COLOR_OPTIONS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 128)
]

font = pygame.font.SysFont('arial', 20)
font_large = pygame.font.SysFont('arial', 30)
font_small = pygame.font.SysFont('arial', 15)
font_tiny = pygame.font.SysFont('arial', 12)

session_data = {}
server_list = []
network_error = None
total_servers = 0
servers_processed = 0

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

def fetch_server_list():
    global network_error, total_servers, servers_processed
    try:
        response = requests.get("https://gkart.pythonanywhere.com/servers", timeout=5)
        response.raise_for_status()
        servers = response.json()
        total_servers = len(servers) * 3  # Each server has 3 endpoints: online, cover, info
        enriched = []
        for srv in servers:
            url = srv["url"]
            start_time = time.time()
            try:
                r = requests.get(f"{url}/online", timeout=2)
                online = r.json().get("online", 0)
            except:
                online = -1
            servers_processed += 1
            ping = int((time.time() - start_time) * 1000)
            cover_surface = None
            description = "No description available"
            try:
                cover_response = requests.get(f"{url}/server_cover", timeout=2)
                if cover_response.status_code == 200:
                    cover_image = BytesIO(cover_response.content)
                    cover_surface = pygame.image.load(cover_image)
                elif cover_response.status_code == 404 and cover_response.json().get('error') == 'Server cover image not found':
                    pass
            except:
                pass
            servers_processed += 1
            try:
                desc_response = requests.get(f"{url}/server_info", timeout=2)
                if desc_response.status_code == 200:
                    description = desc_response.json().get('description', "No description available")
            except:
                pass
            servers_processed += 1
            enriched.append({
                "name": srv["name"],
                "url": url,
                "online": online,
                "ping": ping,
                "cover": cover_surface,
                "description": description,
                "current_height": 200  # Initial height for animation
            })
        return enriched
    except Exception as e:
        network_error = f"Network error: {str(e)}"
        return []

def show_loading_screen():
    global server_list, network_error, total_servers, servers_processed
    loading = True
    progress = 0

    def load():
        nonlocal loading
        server_list.extend(fetch_server_list())
        loading = False

    threading.Thread(target=load).start()

    while loading or (progress < 1.0 and not network_error):
        screen.fill((30, 30, 30))
        # Loading text
        text = font_large.render("Loading servers...", True, (255, 255, 255))
        screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2 - 50))
        
        # Progress bar
        progress_bar_width = 300
        progress_bar_height = 20
        progress_bar_rect = pygame.Rect(WINDOW_WIDTH // 2 - progress_bar_width // 2, WINDOW_HEIGHT // 2, progress_bar_width, progress_bar_height)
        pygame.draw.rect(screen, (100, 100, 100), progress_bar_rect, 2)
        progress = servers_processed / total_servers if total_servers > 0 else 0
        progress_fill = pygame.Rect(progress_bar_rect.x + 2, progress_bar_rect.y + 2, (progress_bar_width - 4) * progress, progress_bar_height - 4)
        pygame.draw.rect(screen, (0, 200, 0), progress_fill)
        
        # Network error
        if network_error:
            error_text = font_small.render(network_error, True, (255, 100, 100))
            screen.blit(error_text, (WINDOW_WIDTH // 2 - error_text.get_width() // 2, WINDOW_HEIGHT // 2 + 50))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

def show_start_screen():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    config = load_config()
    input_name = config['name']
    selected_color = config['color']
    input_active_name = False
    cursor = '_'
    cursor_timer = 0
    cursor_visible = True

    button_width = 100
    button_height = 40
    header_height = 100
    footer_height = 100
    selected_server_height = WINDOW_HEIGHT - header_height - footer_height - 40  # 20px padding top and bottom
    selected_server_width = int(selected_server_height * 16 / 9)  # 16:9 aspect ratio
    other_server_scale = 0.8
    other_server_height = int(selected_server_height * other_server_scale)
    other_server_width = int(other_server_height * 16 / 9)  # 16:9 aspect ratio
    server_spacing = 50
    dot_radius = 5
    dot_spacing = 15

    play_button = pygame.Rect(WINDOW_WIDTH // 2 - button_width // 2, WINDOW_HEIGHT - footer_height + 20, button_width, button_height)
    color_buttons = [
        pygame.Rect(20 + i * 40, 20, 30, 30)
        for i in range(len(COLOR_OPTIONS))
    ]
    name_rect = pygame.Rect(WINDOW_WIDTH - 220, 20, 200, 30)

    selected_server_index = 0
    scroll_position = 0
    scroll_target = 0
    scroll_speed = 0.05  # Slower animation for smoother transitions

    while True:
        screen.fill((50, 50, 50))

        # Title
        title = font_large.render("Karting Game", True, (255, 255, 255))
        screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 20))

        # Name Input (Top Right)
        name_label = font.render("Name:", True, (255, 255, 255))
        screen.blit(name_label, (WINDOW_WIDTH - 260, 25))
        name_surface = font.render(input_name + (cursor if input_active_name and cursor_visible else ''), True, (255, 255, 255))
        pygame.draw.rect(screen, (255, 255, 255), name_rect, 2)
        screen.blit(name_surface, (name_rect.x + 5, name_rect.y + 5))

        # Color Selection (Top Left)
        color_label = font.render("Car Color:", True, (255, 255, 255))
        screen.blit(color_label, (20, 60))
        for i, button in enumerate(color_buttons):
            pygame.draw.rect(screen, COLOR_OPTIONS[i], button)
            if list(selected_color) == list(COLOR_OPTIONS[i]):
                pygame.draw.rect(screen, (255, 255, 255), button, 2)

        # Server Selection (Horizontal Scroll)
        center_x = WINDOW_WIDTH // 2
        for i, srv in enumerate(server_list):
            is_selected = i == selected_server_index
            target_height = selected_server_height if is_selected else other_server_height
            srv['current_height'] += (target_height - srv['current_height']) * scroll_speed
            height = int(srv['current_height'])
            width = int(height * 16 / 9)  # Maintain 16:9 aspect ratio
            offset_x = i * (selected_server_width + server_spacing) - scroll_position
            rect_x = center_x - width // 2 + offset_x
            rect_y = header_height + 20 + (selected_server_height - height) // 2
            rect = pygame.Rect(rect_x, rect_y, width, height)

            # Only draw if the server is at least partially visible
            if rect_x + width >= 0 and rect_x <= WINDOW_WIDTH:
                # Draw server cover or fallback rectangle
                if srv['cover']:
                    scaled_cover = pygame.transform.scale(srv['cover'], (width, height))
                    screen.blit(scaled_cover, (rect_x, rect_y))
                else:
                    pygame.draw.rect(screen, (100, 100, 100), rect)
                if is_selected:
                    pygame.draw.rect(screen, (255, 255, 255), rect, 3)

                # Server Info (Bottom Left)
                name_text = font.render(srv['name'], True, (255, 255, 255))
                screen.blit(name_text, (rect_x + 10, rect_y + height - 90))

                description_segments = parse_description(srv['description'])
                description_lines = wrap_text(description_segments, max_chars=30, max_lines=2)
                render_lines(description_lines, rect_x + 10, rect_y + height - 60, line_spacing=15)

                url_text = font_tiny.render(srv['url'], True, (200, 200, 200))
                screen.blit(url_text, (rect_x + 10, rect_y + height - 20))

                # Ping and Online (Bottom Right)
                ping_text = font.render(f"{srv['ping']}ms", True, (255, 255, 255))
                online_text = font.render(f"Online: {srv['online']}", True, (255, 255, 255))
                screen.blit(ping_text, (rect_x + width - ping_text.get_width() - 10, rect_y + height - 50))
                screen.blit(online_text, (rect_x + width - online_text.get_width() - 10, rect_y + height - 30))

        # Navigation Dots
        if server_list:
            dots_x = WINDOW_WIDTH // 2 - (len(server_list) * dot_spacing - dot_spacing) // 2
            dots_y = WINDOW_HEIGHT - footer_height + 60
            for i in range(len(server_list)):
                color = (255, 255, 255) if i == selected_server_index else (100, 100, 100)
                pygame.draw.circle(screen, color, (dots_x + i * dot_spacing, dots_y), dot_radius)

        # Play Button
        mouse_pos = pygame.mouse.get_pos()
        button_color = BUTTON_HOVER_COLOR if play_button.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(screen, button_color, play_button)
        play_text = font.render("Play", True, (255, 255, 255))
        screen.blit(play_text, (play_button.x + (button_width - play_text.get_width()) // 2,
                                play_button.y + (button_height - play_text.get_height()) // 2))

        pygame.display.flip()

        scroll_position += (scroll_target - scroll_position) * scroll_speed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
                play_button = pygame.Rect(WINDOW_WIDTH // 2 - button_width // 2, WINDOW_HEIGHT - footer_height + 20, button_width, button_height)
                color_buttons = [
                    pygame.Rect(20 + i * 40, 20, 30, 30)
                    for i in range(len(COLOR_OPTIONS))
                ]
                name_rect = pygame.Rect(WINDOW_WIDTH - 220, 20, 200, 30)
                selected_server_height = WINDOW_HEIGHT - header_height - footer_height - 40
                selected_server_width = int(selected_server_height * 16 / 9)
                other_server_height = int(selected_server_height * other_server_scale)
                other_server_width = int(other_server_height * 16 / 9)
                scroll_target = selected_server_index * (selected_server_width + server_spacing)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if play_button.collidepoint(event.pos):
                        if server_list:
                            selected_server_url = server_list[selected_server_index]['url']
                            save_config(input_name, selected_color, selected_server_url)
                            return input_name, selected_color, selected_server_url
                    if name_rect.collidepoint(event.pos):
                        input_active_name = True
                    else:
                        input_active_name = False

                    for i, srv in enumerate(server_list):
                        height = int(srv['current_height'])
                        width = int(height * 16 / 9)
                        offset_x = i * (selected_server_width + server_spacing) - scroll_position
                        rect_x = center_x - width // 2 + offset_x
                        rect_y = header_height + 20 + (selected_server_height - height) // 2
                        rect = pygame.Rect(rect_x, rect_y, width, height)
                        if rect.collidepoint(event.pos):
                            selected_server_index = i
                            scroll_target = i * (selected_server_width + server_spacing)

                elif event.button == 4:  # Scroll up (left)
                    if selected_server_index > 0:
                        selected_server_index -= 1
                        scroll_target = selected_server_index * (selected_server_width + server_spacing)
                elif event.button == 5:  # Scroll down (right)
                    if selected_server_index < len(server_list) - 1:
                        selected_server_index += 1
                        scroll_target = selected_server_index * (selected_server_width + server_spacing)

            elif event.type == pygame.KEYDOWN:
                if input_active_name:
                    if event.key == pygame.K_BACKSPACE:
                        input_name = input_name[:-1]
                    elif event.key == pygame.K_RETURN:
                        input_active_name = False
                    elif event.unicode.isalnum() or event.unicode in [' ', '.', ':']:
                        if len(input_name) < 20:
                            input_name += event.unicode
                else:
                    if event.key == pygame.K_LEFT and selected_server_index > 0:
                        selected_server_index -= 1
                        scroll_target = selected_server_index * (selected_server_width + server_spacing)
                    elif event.key == pygame.K_RIGHT and selected_server_index < len(server_list) - 1:
                        selected_server_index += 1
                        scroll_target = selected_server_index * (selected_server_width + server_spacing)

        cursor_timer += 1
        if cursor_timer >= 30:
            cursor_visible = not cursor_visible
            cursor_timer = 0

if __name__ == "__main__":
    show_loading_screen()
    while True:
        player_name, player_color, server_url = show_start_screen()
        server_url2 = server_url[8:]
        max_attempts = 2
        for attempt in [1]:
            local_car, camera, network_thread_obj = attempt_game_start(player_name, player_color, server_url2)
            if local_car is None:
                continue
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