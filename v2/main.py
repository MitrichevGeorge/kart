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

session_data = {}
server_list = []

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
    try:
        response = requests.get("https://gkart.pythonanywhere.com/servers", timeout=5)
        servers = response.json()
        enriched = []
        for srv in servers:
            url = srv["url"]
            start_time = time.time()
            try:
                r = requests.get(f"{url}/online", timeout=2)
                online = r.json().get("online", 0)
            except:
                online = -1
            ping = int((time.time() - start_time) * 1000)
            enriched.append({"name": srv["name"], "url": url, "online": online, "ping": ping})
        return enriched
    except Exception as e:
        print("Error fetching server list:", e)
        return []

def show_loading_screen():
    global server_list
    loading = True

    def load():
        nonlocal loading
        server_list.extend(fetch_server_list())
        loading = False

    threading.Thread(target=load).start()

    while loading:
        screen.fill((30, 30, 30))
        text = font_large.render("Loading servers...", True, (255, 255, 255))
        screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2 - 20))
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

    selected_server_index = 0
    scroll_offset = 0
    max_visible_servers = 5

    while True:
        play_button = pygame.Rect(WINDOW_WIDTH // 2 - button_width // 2, WINDOW_HEIGHT - 100, button_width, button_height)
        color_buttons = [
            pygame.Rect(WINDOW_WIDTH // 2 - len(COLOR_OPTIONS) * 40 // 2 + i * 40, WINDOW_HEIGHT // 2 + 150, 30, 30)
            for i in range(len(COLOR_OPTIONS))
        ]
        name_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 120, 200, 30)
        server_list_area = pygame.Rect(WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 60, 300, 150)

        screen.fill((50, 50, 50))

        title = font_large.render("Karting Game Setup", True, (255, 255, 255))
        screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 50))

        # Name Input
        name_label = font.render("Name:", True, (255, 255, 255))
        screen.blit(name_label, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 120))
        name_surface = font.render(input_name + (cursor if input_active_name and cursor_visible else ''), True, (255, 255, 255))
        pygame.draw.rect(screen, (255, 255, 255), name_rect, 2)
        screen.blit(name_surface, (name_rect.x + 5, name_rect.y + 5))

        # Server List
        server_label = font.render("Select Server:", True, (255, 255, 255))
        screen.blit(server_label, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 90))
        pygame.draw.rect(screen, (100, 100, 100), server_list_area)

        visible_servers = server_list[scroll_offset:scroll_offset + max_visible_servers]
        for i, srv in enumerate(visible_servers):
            rect = pygame.Rect(server_list_area.x, server_list_area.y + i * 30, 300, 30)
            is_selected = selected_server_index == i + scroll_offset
            color = (80, 80, 120) if is_selected else (60, 60, 60)
            pygame.draw.rect(screen, color, rect)
            info = f"{srv['name']} | Online: {srv['online']} | {srv['ping']}ms"
            text = font_small.render(info, True, (255, 255, 255))
            screen.blit(text, (rect.x + 5, rect.y + 5))

        # Color Selection
        color_label = font.render("Car Color:", True, (255, 255, 255))
        screen.blit(color_label, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 + 110))
        for i, button in enumerate(color_buttons):
            pygame.draw.rect(screen, COLOR_OPTIONS[i], button)
            if list(selected_color) == list(COLOR_OPTIONS[i]):
                pygame.draw.rect(screen, (255, 255, 255), button, 2)

        # Play Button
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

                    for i in range(len(visible_servers)):
                        rect = pygame.Rect(server_list_area.x, server_list_area.y + i * 30, 300, 30)
                        if rect.collidepoint(event.pos):
                            selected_server_index = scroll_offset + i

                    for i, button in enumerate(color_buttons):
                        if button.collidepoint(event.pos):
                            selected_color = COLOR_OPTIONS[i]

                elif event.button == 4:  # Scroll up
                    if scroll_offset > 0:
                        scroll_offset -= 1
                elif event.button == 5:  # Scroll down
                    if scroll_offset < max(0, len(server_list) - max_visible_servers):
                        scroll_offset += 1

            elif event.type == pygame.KEYDOWN:
                if input_active_name:
                    if event.key == pygame.K_BACKSPACE:
                        input_name = input_name[:-1]
                    elif event.key == pygame.K_RETURN:
                        input_active_name = False
                    elif event.unicode.isalnum() or event.unicode in [' ', '.', ':']:
                        if len(input_name) < 20:
                            input_name += event.unicode

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