import pygame
import sys
import math

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
CAR_COLOR = (128, 128, 128)
CAR2_COLOR = (255, 0, 0)  # Different color for second car
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
SAND_SLOWDOWN = 0.1
WALL_BOUNCE = 0.3
FRICTION = 0.3
TRAIL_FADE_RATE = 0.99
MIN_SPEED_FOR_TURN = 0.5
LOW_SPEED_TURN_FACTOR = 0.3
HIGH_SPEED_DRIFT_FACTOR = 0.3
CAR_COLLISION_BOUNCE = 0.5  # Bounce factor for car collisions
MIN_SPAWN_DISTANCE = 30  # Minimum distance between cars at spawn

# Поверхность для следов
trail_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)

# Шрифт
font = pygame.font.SysFont('arial', 20)

class Car:
    def __init__(self, x, y, angle, render_enabled=True, training_mode=False, is_player_one=True):
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
        self.next_checkpoint = 0
        self.total_reward = 0
        self.is_player_one = is_player_one  # Flag to determine control scheme

        self.wheel_positions = [
            (-CAR_WIDTH // 2 + 5, CAR_HEIGHT // 2),
            (-CAR_WIDTH // 2 + 5, -CAR_HEIGHT // 2),
            (CAR_WIDTH // 2 - 5, CAR_HEIGHT // 2),
            (CAR_WIDTH // 2 - 5, -CAR_HEIGHT // 2)
        ]

    def update(self, keys):
        # Cache trigonometric values
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        surface_color = get_surface_color(self.x, self.y)

        accel = 0
        turn_input = 0
        speed_factor = abs(self.speed) / MAX_SPEED

        # Player 1 uses arrow keys, Player 2 uses WASD
        if self.is_player_one:
            self.is_accelerating = keys[pygame.K_UP]
            self.is_braking = keys[pygame.K_DOWN]
            self.is_turning_left = keys[pygame.K_LEFT]
            self.is_turning_right = keys[pygame.K_RIGHT]
        else:
            self.is_accelerating = keys[pygame.K_w]
            self.is_braking = keys[pygame.K_s]
            self.is_turning_left = keys[pygame.K_a]
            self.is_turning_right = keys[pygame.K_d]

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
        # Use different color for each player
        color = CAR_COLOR if self.is_player_one else CAR2_COLOR
        pygame.draw.polygon(screen, color, rotated_points)

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

        # Draw checkpoint count above car
        checkpoint_text = font.render(f"CP: {self.checkpoints_passed}", True, (0, 255, 0))
        screen.blit(checkpoint_text, (self.x - CAR_WIDTH // 2, self.y - CAR_HEIGHT - 60))

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
        self.next_checkpoint = 0
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

def find_safe_spawn_positions():
    start_x, start_y = find_start_position()
    positions = []
    attempts = 0
    max_attempts = 50
    search_radius = 10

    # Try to place first car at start position
    if get_surface_color(start_x, start_y) != COLOR_WALL:
        positions.append((start_x, start_y))
    else:
        # Find nearby valid position
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                new_x = start_x + dx
                new_y = start_y + dy
                if get_surface_color(new_x, new_y) not in [COLOR_WALL, COLOR_SAND]:
                    positions.append((new_x, new_y))
                    break
            if positions:
                break

    # Find position for second car
    while len(positions) < 2 and attempts < max_attempts:
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                new_x = start_x + dx
                new_y = start_y + dy
                if get_surface_color(new_x, new_y) not in [COLOR_WALL, COLOR_SAND]:
                    # Check distance from first car
                    too_close = False
                    for pos_x, pos_y in positions:
                        distance = math.sqrt((new_x - pos_x)**2 + (new_y - pos_y)**2)
                        if distance < MIN_SPAWN_DISTANCE:
                            too_close = True
                            break
                    if not too_close:
                        positions.append((new_x, new_y))
                        return positions
        search_radius += 5
        attempts += 1

    # Fallback: use default positions if no valid ones found
    if len(positions) < 2:
        positions.append((start_x + MIN_SPAWN_DISTANCE, start_y))
    return positions

def check_collision(car1, car2):
    # Simple rectangular collision detection
    rect1 = pygame.Rect(car1.x - CAR_WIDTH // 2, car1.y - CAR_HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT)
    rect2 = pygame.Rect(car2.x - CAR_WIDTH // 2, car2.y - CAR_HEIGHT // 2, CAR_WIDTH, CAR_HEIGHT)
    
    if rect1.colliderect(rect2):
        # Calculate collision response
        dx = car1.x - car2.x
        dy = car1.y - car2.y
        distance = max(math.sqrt(dx**2 + dy**2), 0.1)  # Avoid division by zero
        
        # Normalize direction
        nx = dx / distance
        ny = dy / distance
        
        # Relative velocity
        rvx = car1.velocity_x - car2.velocity_x
        rvy = car1.velocity_y - car2.velocity_y
        
        # Dot product
        dot = rvx * nx + rvy * ny
        
        if dot > 0:  # Cars are moving towards each other
            # Impulse calculation
            impulse = dot / 2  # Simple mass assumption (equal mass)
            
            # Apply impulse
            car1.velocity_x -= impulse * nx * CAR_COLLISION_BOUNCE
            car1.velocity_y -= impulse * ny * CAR_COLLISION_BOUNCE
            car2.velocity_x += impulse * nx * CAR_COLLISION_BOUNCE
            car2.velocity_y += impulse * ny * CAR_COLLISION_BOUNCE
            
            # Adjust speeds
            car1.speed *= CAR_COLLISION_BOUNCE
            car2.speed *= CAR_COLLISION_BOUNCE
            car1.angular_velocity *= CAR_COLLISION_BOUNCE
            car2.angular_velocity *= CAR_COLLISION_BOUNCE
            
            # Push cars apart to prevent sticking
            overlap = (CAR_WIDTH + CAR_HEIGHT) / 2 - distance
            if overlap > 0:
                car1.x += nx * overlap / 2
                car1.y += ny * overlap / 2
                car2.x -= nx * overlap / 2
                car2.y -= ny * overlap / 2

# Initialize two cars
spawn_positions = find_safe_spawn_positions()
car1 = Car(spawn_positions[0][0], spawn_positions[0][1], 0, is_player_one=True)
car2 = Car(spawn_positions[1][0], spawn_positions[1][1], 0, is_player_one=False)

def main():
    clock = pygame.time.Clock()
    FPS = 60
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        car1.update(keys)
        car2.update(keys)
        
        # Check for collision between cars
        check_collision(car1, car2)

        faded_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)
        faded_surface.blit(trail_surface, (0, 0))
        faded_surface.set_alpha(int(255 * TRAIL_FADE_RATE))
        trail_surface.fill((0, 0, 0, 0))
        trail_surface.blit(faded_surface, (0, 0))

        screen.blit(map_image, (0, 0))
        screen.blit(trail_surface, (0, 0))
        car1.draw()
        car2.draw()
        pygame.display.flip()

        clock.tick(FPS)

if __name__ == "__main__":
    main()