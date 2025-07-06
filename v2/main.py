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

pygame.init()

WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Karting Game")

setup_screen(screen)

font = pygame.font.SysFont('arial', 20)
font_large = pygame.font.SysFont('arial', 30)
font_small = pygame.font.SysFont('arial', 15)

session_data = {}

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

def show_start_screen():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    config = load_config()
    input_name = config['name']
    selected_color = config['color']
    input_server_url = config['server_url']
    input_active_name = False
    input_active_url = False
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
        name_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 80, 200, 30)
        url_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 20, 200, 30)

        screen.fill((50, 50, 50))
        
        title = font_large.render("Karting Game Setup", True, (255, 255, 255))
        screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 50))

        name_label = font.render("Name:", True, (255, 255, 255))
        screen.blit(name_label, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 80))
        name_surface = font.render(input_name + (cursor if input_active_name and cursor_visible else ''), True, (255, 255, 255))
        pygame.draw.rect(screen, (255, 255, 255), name_rect, 2)
        screen.blit(name_surface, (name_rect.x + 5, name_rect.y + 5))

        url_label = font.render("URL:", True, (255, 255, 255))
        screen.blit(url_label, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 20))
        url_surface = font.render(input_server_url + (cursor if input_active_url and cursor_visible else ''), True, (255, 255, 255))
        pygame.draw.rect(screen, (255, 255, 255), url_rect, 2)
        screen.blit(url_surface, (url_rect.x + 5, url_rect.y + 5))

        color_label = font.render("Car Color:", True, (255, 255, 255))
        screen.blit(color_label, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 + 20))
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
                    save_config(input_name, selected_color, input_server_url)
                    return input_name, selected_color, input_server_url
                if name_rect.collidepoint(event.pos):
                    input_active_name = True
                    input_active_url = False
                elif url_rect.collidepoint(event.pos):
                    input_active_url = True
                    input_active_name = False
                else:
                    input_active_name = False
                    input_active_url = False
                for i, button in enumerate(color_buttons):
                    if button.collidepoint(event.pos):
                        selected_color = COLOR_OPTIONS[i]
            elif event.type == pygame.KEYDOWN:
                if input_active_name:
                    if event.key == pygame.K_BACKSPACE:
                        input_name = input_name[:-1]
                    elif event.key == pygame.K_RETURN:
                        input_active_name = False
                    elif event.unicode.isalnum() or event.unicode in [' ', '.', ':']:
                        if len(input_name) < 20:
                            input_name += event.unicode
                elif input_active_url:
                    if event.key == pygame.K_BACKSPACE:
                        input_server_url = input_server_url[:-1]
                    elif event.key == pygame.K_RETURN: 
                        input_active_url = False
                    elif event.unicode.isalnum() or event.unicode in ['.', ':']:
                        if len(input_server_url) < 50:
                            input_server_url += event.unicode

        cursor_timer += 1
        if cursor_timer >= 30:
            cursor_visible = not cursor_visible
            cursor_timer = 0

if __name__ == "__main__":
    while True:
        player_name, player_color, server_url = show_start_screen()
        server_url2 = server_url + ".pythonanywhere.com"
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
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