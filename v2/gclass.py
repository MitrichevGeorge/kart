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

PHYSICS_PARAMS = {}
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

def init(params):
    global PHYSICS_PARAMS
    PHYSICS_PARAMS = params

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
        self.lap_count = 1
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
        self.health = PHYSICS_PARAMS.get('MAX_HEALTH', 20)
        self.nitro = PHYSICS_PARAMS.get('NITRO_MAX', 100)
        self.spawn_protection = True
        self.spawn_time = time.time()
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
                        lap_time = time.time() - current_lap_start
                        lap_times.append(lap_time)
                        session_data['best_lap_times'][PLAYER_ID] = min(session_data['best_lap_times'].get(PLAYER_ID, float('inf')), lap_time)
                        session_data['total_laps'][PLAYER_ID] = self.lap_count - 1 + self.checkpoints_passed / total_checkpoints
                        save_session_data()
                    self.lap_count += 1
                    self.checkpoints_passed = 0
                    if self.is_local_player:
                        current_lap_start = time.time()
                elif self.is_local_player and self.checkpoints_passed == 1 and self.lap_count == 1:
                    current_lap_start = time.time()
                break

    def update(self, keys=None):
        if self.is_dead and (time.time() - self.death_time > PHYSICS_PARAMS['CORPSE_LIFETIME']):
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
        speed_factor = abs(self.speed) / PHYSICS_PARAMS['MAX_SPEED']
        if self.is_local_player and keys:
            self.is_accelerating = keys[pygame.K_UP] or keys[pygame.K_w]
            self.is_braking = keys[pygame.K_DOWN] or keys[pygame.K_s]
            self.is_turning_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
            self.is_turning_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
            self.is_drifting = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            self.is_using_nitro = (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]) and (self.is_accelerating or self.is_braking)
        if self.is_accelerating:
            accel = PHYSICS_PARAMS['ACCELERATION']
        if self.is_braking:
            accel = -PHYSICS_PARAMS['ACCELERATION'] * 0.5
        if self.is_using_nitro and self.nitro > 0:
            accel *= PHYSICS_PARAMS['NITRO_BOOST_FACTOR']
            self.nitro = max(0, self.nitro - PHYSICS_PARAMS['NITRO_CONSUMPTION_RATE'] * delta_time)
        else:
            self.nitro = min(PHYSICS_PARAMS['NITRO_MAX'], self.nitro + PHYSICS_PARAMS['NITRO_REGEN_RATE'] * delta_time)
        if self.nitro <= PHYSICS_PARAMS['NITRO_LOW_THRESHOLD']:
            self.speed *= PHYSICS_PARAMS['NITRO_LOW_SLOWDOWN']
            if not self.is_dead:
                damage = PHYSICS_PARAMS['NITRO_LOW_DAMAGE'] * delta_time
                self.health = max(0, self.health - damage)
                if damage > 0.1:
                    self.damage_popups.append(DamagePopup(self.x, self.y, damage))
        if self.is_turning_left:
            turn_input = -PHYSICS_PARAMS['TURN_ACCELERATION']
            self.steering_angle = max(self.steering_angle - PHYSICS_PARAMS['TURN_ACCELERATION'], -math.pi / 6)
        elif self.is_turning_right:
            turn_input = PHYSICS_PARAMS['TURN_ACCELERATION']
            self.steering_angle = min(self.steering_angle + PHYSICS_PARAMS['TURN_ACCELERATION'], math.pi / 6)
        else:
            self.steering_angle *= 0.8
        if surface_color == COLOR_SAND:
            accel *= PHYSICS_PARAMS['SAND_SLOWDOWN']
            turn_input *= PHYSICS_PARAMS['SAND_SLOWDOWN']
            self.speed *= (1 - PHYSICS_PARAMS['SAND_INERTIA_LOSS'])
            self.velocity_x *= (1 - PHYSICS_PARAMS['SAND_INERTIA_LOSS'])
            self.velocity_y *= (1 - PHYSICS_PARAMS['SAND_INERTIA_LOSS'])
            self.angular_velocity *= (1 - PHYSICS_PARAMS['SAND_INERTIA_LOSS'])
        self.last_speed = self.speed
        self.speed += accel
        self.speed = max(min(self.speed, PHYSICS_PARAMS['MAX_SPEED']), -PHYSICS_PARAMS['MAX_SPEED'] / 2)
        if abs(self.speed) < PHYSICS_PARAMS['DECELERATION'] and not self.is_accelerating and not self.is_braking:
            self.speed = 0
        else:
            self.speed *= (1 - PHYSICS_PARAMS['DECELERATION'])
        if abs(self.speed) > PHYSICS_PARAMS['MIN_SPEED_FOR_TURN']:
            turn_scale = PHYSICS_PARAMS['LOW_SPEED_TURN_FACTOR'] + (1 - PHYSICS_PARAMS['LOW_SPEED_TURN_FACTOR']) * speed_factor
            self.angular_velocity += turn_input * turn_scale
            max_angular = PHYSICS_PARAMS['MAX_ANGULAR_VELOCITY'] * turn_scale
            self.angular_velocity = max(min(self.angular_velocity, max_angular), -max_angular)
        else:
            self.angular_velocity *= 0.5
        if not self.is_turning_left and not self.is_turning_right:
            self.angular_velocity *= (1 - PHYSICS_PARAMS['ROTATIONAL_FRICTION'])
        self.angle += self.angular_velocity * (1 - speed_factor * 0.5)
        drift_factor = PHYSICS_PARAMS['DRIFT_FACTOR_ON_SHIFT'] if self.is_drifting else PHYSICS_PARAMS['HIGH_SPEED_DRIFT_FACTOR']
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
        nsc = get_surface_color(new_x, new_y)
        if nsc == COLOR_WALL:
            old_velocity = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
            self.velocity_x *= -PHYSICS_PARAMS['WALL_BOUNCE']
            self.velocity_y *= -PHYSICS_PARAMS['WALL_BOUNCE']
            new_velocity = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
            impulse = old_velocity - new_velocity
            damage = impulse * PHYSICS_PARAMS['DAMAGE_SCALING']
            if damage > 0.1:
                self.damage_popups.append(DamagePopup(self.x, self.y, damage))
            self.health = max(0, self.health - damage)
            self.speed *= PHYSICS_PARAMS['WALL_BOUNCE']
            self.angular_velocity *= PHYSICS_PARAMS['WALL_BOUNCE']
        elif nsc == COLOR_SPRING_WALL:
            old_velocity = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
            if old_velocity > 0.001:
                normal_x = -self.velocity_x / old_velocity
                normal_y = -self.velocity_y / old_velocity
            else:
                normal_x, normal_y = 0, -1
            bounce_factor = PHYSICS_PARAMS['WALL_BOUNCE'] * 2.0
            self.velocity_x = normal_x * old_velocity * bounce_factor
            self.velocity_y = normal_y * old_velocity * bounce_factor
            new_velocity = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
            impulse = old_velocity + new_velocity
            damage = impulse * PHYSICS_PARAMS['DAMAGE_SCALING'] * 0.1
            self.velocity_x *= PHYSICS_PARAMS['WALL_BOUNCE'] * 10
            self.velocity_y *= PHYSICS_PARAMS['WALL_BOUNCE'] * 10
            if damage > 0.1:
                self.damage_popups.append(DamagePopup(self.x, self.y, damage))
            self.health = max(0, self.health - damage)
            self.speed = new_velocity
            self.angular_velocity *= PHYSICS_PARAMS['WALL_BOUNCE'] * 0.5
            current_time = time.time()
            if current_time - self.last_spark_time >= PHYSICS_PARAMS['SPARK_EMISSION_RATE']:
                self.spark_particles.append(SparkParticle(self.x, self.y))
                self.last_spark_time = current_time
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
        if (distance_from_spawn > PHYSICS_PARAMS['MIN_SPAWN_DISTANCE'] or
            time.time() - self.spawn_time > PHYSICS_PARAMS['SPAWN_PROTECTION_TIME']):
            self.spawn_protection = False
        if self.health <= PHYSICS_PARAMS['SMOKE_HEALTH_THRESHOLD'] and not self.training_mode and not self.is_dead:
            current_time = time.time()
            if current_time - self.last_smoke_time >= PHYSICS_PARAMS['SMOKE_EMISSION_RATE']:
                self.smoke_particles.append(SmokeParticle(self.x, self.y))
                self.last_smoke_time = current_time
        if self.is_using_nitro and self.nitro > 0 and not self.training_mode and not self.is_dead:
            current_time = time.time()
            if current_time - self.last_nitro_flame_time >= PHYSICS_PARAMS['NITRO_FLAME_EMISSION_RATE']:
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
        trail_alpha = min(int(relative_speed / PHYSICS_PARAMS['MAX_SPEED'] * 255 * PHYSICS_PARAMS['FRICTION'] * 2), 255)
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        surface_color = get_surface_color(self.x, self.y)
        current_time = time.time()
        for i, (wx, wy) in enumerate(self.wheel_positions):
            wheel_x = self.x + wx * cos_angle - wy * sin_angle
            wheel_y = self.y + wx * sin_angle + wy * cos_angle
            adjusted_trail_alpha = trail_alpha
            if i >= 2 and abs(self.steering_angle) > 0.01:
                adjusted_trail_alpha = min(trail_alpha * 1.5, 255)
            if adjusted_trail_alpha > 5:
                pygame.draw.circle(trail_surface, (*TRAIL_COLOR, adjusted_trail_alpha), (int(wheel_x), int(wheel_y)), 3)
            if (self.health <= PHYSICS_PARAMS['SMOKE_HEALTH_THRESHOLD'] and not self.training_mode and not self.is_dead and
                surface_color == COLOR_FLOOR and adjusted_trail_alpha > PHYSICS_PARAMS['SPARK_ALPHA_THRESHOLD'] and
                current_time - self.last_spark_time >= PHYSICS_PARAMS['SPARK_EMISSION_RATE']):
                self.spark_particles.append(SparkParticle(wheel_x, wheel_y))
                self.last_spark_time = current_time

    def draw(self, camera):
        if not self.render_enabled or self.training_mode or (self.is_dead and (time.time() - self.death_time > PHYSICS_PARAMS['CORPSE_LIFETIME'])):
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
                wheel_angle = self.angle + (self.steering_angle if i >= 2 else 0)
                wheel_h = WHEEL_HEIGHT if i < 2 else FRONT_WHEEL_HEIGHT
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
                    arrow_start = (
                        car_screen_pos[0] + norm_dx * ARROW_OFFSET * camera.zoom,
                        car_screen_pos[1] + norm_dy * ARROW_OFFSET * camera.zoom
                    )
                    arrow_end = (
                        arrow_start[0] + norm_dx * ARROW_LENGTH * camera.zoom,
                        arrow_start[1] + norm_dy * ARROW_LENGTH * camera.zoom
                    )
                    arrow_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.line(arrow_surface, ARROW_COLOR, arrow_start, arrow_end, ARROW_THICKNESS)
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
            render_text_with_outline(self.name, font, (255, 255, 255), (self.x - CAR_WIDTH // 2, self.y - CAR_HEIGHT - PHYSICS_PARAMS['NAME_OFFSET']), camera)
            draw_health_bar(screen, camera, self.x, self.y, self.health, PHYSICS_PARAMS['MAX_HEALTH'])
            if self.nitro < PHYSICS_PARAMS['NITRO_MAX'] * PHYSICS_PARAMS['NITRO_VISIBILITY_THRESHOLD']:
                draw_nitro_bar(screen, camera, self.x, self.y, self.nitro, PHYSICS_PARAMS['NITRO_MAX'])
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
        self.health = PHYSICS_PARAMS['MAX_HEALTH']
        self.nitro = PHYSICS_PARAMS['NITRO_MAX']
        self.spawn_protection = True
        self.spawn_time = time.time()
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