import pygame
import sys
import requests
from io import BytesIO

pygame.init()

WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Karting Game")

COLOR_FLOOR = (0, 0, 0)
COLOR_WALL = (200, 200, 200)
COLOR_START = (50, 200, 0)

CAR_WIDTH = 25
CAR_HEIGHT = 20
CAR_COLOR = (255, 0, 0)

map_image = None
MAP_WIDTH = 0
MAP_HEIGHT = 0
SERVER_URL = "geomit25.pythonanywhere.com"

def load_map():
    global map_image, MAP_WIDTH, MAP_HEIGHT
    try:
        map_response = requests.get(f'http://{SERVER_URL}/map', timeout=5)
        if map_response.status_code == 200:
            map_data = BytesIO(map_response.content)
            map_image = pygame.image.load(map_data)
            MAP_WIDTH, MAP_HEIGHT = map_image.get_size()
        else:
            print("Failed to load map from server")
            return False
    except Exception as e:
        print(f"Error loading map: {e}")
        return False
    return True

def get_surface_color(x, y):
    if map_image is None or x < 0 or x >= MAP_WIDTH or y < 0 or y >= MAP_HEIGHT:
        return COLOR_WALL
    return map_image.get_at((int(x), int(y)))[:3]

def find_start_position():
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            if map_image.get_at((x, y))[:3] == COLOR_START:
                return x, y
    return MAP_WIDTH // 2, MAP_HEIGHT // 2

class Camera:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.zoom = 1.0

    def update(self, target_x, target_y):
        self.x = target_x
        self.y = target_y

    def apply_transform(self, pos):
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

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 5

    def update(self, keys):
        new_x = self.x
        new_y = self.y

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            new_y -= self.speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            new_y += self.speed
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            new_x -= self.speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            new_x += self.speed

        if get_surface_color(new_x, new_y) != COLOR_WALL:
            self.x = new_x
            self.y = new_y

    def draw(self, camera):
        points = [
            (-CAR_WIDTH // 2, -CAR_HEIGHT // 2),
            (CAR_WIDTH // 2, -CAR_HEIGHT // 2),
            (CAR_WIDTH // 2, CAR_HEIGHT // 2),
            (-CAR_WIDTH // 2, CAR_HEIGHT // 2)
        ]
        rotated_points = []
        for x, y in points:
            screen_pos = camera.apply_transform((self.x + x, self.y + y))
            rotated_points.append(screen_pos)
        pygame.draw.polygon(screen, CAR_COLOR, rotated_points)

def main():
    global screen, WINDOW_WIDTH, WINDOW_HEIGHT
    if not load_map():
        print("Exiting due to map load failure")
        pygame.quit()
        sys.exit()

    start_x, start_y = find_start_position()
    car = Car(start_x, start_y)
    camera = Camera()
    camera.x = start_x
    camera.y = start_y

    clock = pygame.time.Clock()
    FPS = 60

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)

        keys = pygame.key.get_pressed()
        car.update(keys)
        camera.update(car.x, car.y)

        screen.fill((0, 0, 0))

        visible_width = int(WINDOW_WIDTH / camera.zoom)
        visible_height = int(WINDOW_HEIGHT / camera.zoom)
        visible_x = int(camera.x - visible_width / 2)
        visible_y = int(camera.y - visible_height / 2)
        visible_x = max(0, min(visible_x, MAP_WIDTH - visible_width))
        visible_y = max(0, min(visible_y, MAP_HEIGHT - visible_y))
        visible_width = min(visible_width, MAP_WIDTH - visible_x)
        visible_height = min(visible_height, MAP_HEIGHT - visible_y)

        map_rect = pygame.Rect(visible_x, visible_y, visible_width, visible_height)
        cropped_map = map_image.subsurface(map_rect)
        scaled_map, map_pos = camera.apply_surface_transform(cropped_map, (visible_x, visible_y))
        screen.blit(scaled_map, map_pos)

        car.draw(camera)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()